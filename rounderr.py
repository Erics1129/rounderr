#!/usr/bin/env python3
"""
rounderr.py   reverse mode rounding error interpreter
======================================================

Type a formula (plain, LaTeX, or a line by line tape), give numbers, and get

  * the tape                one line per rounded floating point step
  * lambda_i                the adjoint (blame weight) of every line, from ONE backward sweep
  * what actually happened  actual error = computed minus a 60 digit reference
  * what the error might be first order prediction, worst case ceiling, typical size
  * who is to blame         per line weights |lambda_i d_i t_i|, ranked
  * why and what to do      cancellation / absorption diagnosis, rewrite candidates
  * a LaTeX report          --tex report.tex  [--pdf]

The math (Bindel, FPTalks 2026 notation, which is also the whiteboard notation)

  tape         F(t, x) = t - G(t, x) = 0,          y = w^T t        (w picks the output line)
  perturbed    F(t_hat, x_hat) = rho,                rho_i = local rounding slip of line i
  adjoint      F_t^T lambda = w                      (back substitution = the reverse sweep)
  forward err  Delta y  =  lambda^T rho              (Linnainmaa 1976, first order Taylor)
  ceiling      |Delta y| <= u * sum_i |lambda_i d_i t_i|          worst case, |rho_i| <= d_i |t_i| u
  typical      u * ||lambda d t||_2 / sqrt(3)                       iid uniform slips
  gradient     grad_x y = G_x^T lambda                              (= lambda on the input lines)
  conditioning kappa  = sum_j |g_j x_j| / |y|                       what the PROBLEM forces
  amplification A     = sum_i |lambda_i d_i t_i| / |y|              what the ALGORITHM allows
  verdict      stable when A <= max(10, number of rounded steps) * max(1, kappa)
  backward     beta_1 = ||v||_1 / (||g||_2 ||x||_2),  v_i = lambda_i d_i t_i   (Bindel)

Needs Python 3.8+.  mpmath gives the 60 digit reference for sqrt, exp, sin ...
(pip install mpmath).  Without it the program falls back to decimal for
sqrt / exp / log / log10 and refuses the trig functions.
"""

import argparse
import ast
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from fractions import Fraction

# ----------------------------------------------------------------------------
# 0.  high precision engine (mpmath if present, decimal fallback)
# ----------------------------------------------------------------------------
try:
    import mpmath
    mp = mpmath.mp
    mp.dps = 60
    HAVE_MP = True
except ImportError:  # pragma: no cover
    mpmath = None
    HAVE_MP = False
    import decimal
    decimal.getcontext().prec = 60


def hp(x):
    """exact float or Fraction  ->  high precision number (mpf or Decimal)"""
    fr = Fraction(x)
    if HAVE_MP:
        return mpmath.mpf(fr.numerator) / mpmath.mpf(fr.denominator)
    return decimal.Decimal(fr.numerator) / decimal.Decimal(fr.denominator)


def hp_to_fraction(v):
    """high precision number -> exact Fraction (used when emulating a precision)"""
    if isinstance(v, Fraction):
        return v
    if HAVE_MP:
        if not mpmath.isfinite(v):
            raise OverflowError("non finite value")
        sign, man, exp, _ = v._mpf_
        fr = Fraction(man) * (Fraction(2) ** exp)
        return -fr if sign else fr
    return Fraction(v)  # Decimal -> Fraction is exact


def hp_float(v):
    return float(v)


# high precision versions of the library functions
def hp_func(name, a, b=None):
    if HAVE_MP:
        table = {
            'sqrt': mpmath.sqrt, 'exp': mpmath.exp, 'log': mpmath.log,
            'log10': mpmath.log10, 'log2': lambda z: mpmath.log(z, 2),
            'sin': mpmath.sin, 'cos': mpmath.cos, 'tan': mpmath.tan,
            'asin': mpmath.asin, 'acos': mpmath.acos, 'atan': mpmath.atan,
            'sinh': mpmath.sinh, 'cosh': mpmath.cosh, 'tanh': mpmath.tanh,
            'expm1': mpmath.expm1, 'log1p': mpmath.log1p,
        }
        if name == 'pow':
            return mpmath.power(a, b)
        return table[name](a)
    # decimal fallback (no trig)
    table = {
        'sqrt': lambda z: z.sqrt(), 'exp': lambda z: z.exp(), 'log': lambda z: z.ln(),
        'log10': lambda z: z.log10(), 'log2': lambda z: z.ln() / decimal.Decimal(2).ln(),
        'expm1': lambda z: z.exp() - 1, 'log1p': lambda z: (1 + z).ln(),
    }
    if name == 'pow':
        return (b * a.ln()).exp()
    if name not in table:
        raise RuntimeError("function %s needs mpmath (pip install mpmath)" % name)
    return table[name](a)


# ----------------------------------------------------------------------------
# 1.  THE READER   (text -> python expression, LLM style: tolerant + explains guesses)
# ----------------------------------------------------------------------------
FUNCS = {'sqrt', 'exp', 'log', 'log10', 'log2', 'sin', 'cos', 'tan', 'asin', 'acos',
         'atan', 'sinh', 'cosh', 'tanh', 'abs', 'expm1', 'log1p', 'pow'}
ALIASES = {'ln': 'log', 'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan', 'fabs': 'abs'}
CONSTS = {'pi', 'e'}
KEEP = {'alpha', 'beta', 'gamma', 'delta', 'eps', 'epsilon', 'theta', 'phi', 'psi', 'mu', 'nu',
        'tau', 'rho', 'sigma', 'omega', 'lam', 'kappa', 'eta', 'zeta', 'xi', 'chi', 'dt', 'dx', 'dy'}
LATEX_WORD = {'sin', 'cos', 'tan', 'exp', 'ln', 'log', 'arcsin', 'arccos', 'arctan',
              'sinh', 'cosh', 'tanh', 'sqrt', 'pi'}
LATEX_MAP = {'cdot': '*', 'times': '*', 'div': '/'}
LATEX_DROP = {'left', 'right', ',', ';', '!', ' ', 'quad', 'qquad', 'displaystyle',
              'mathrm', 'operatorname', 'text', 'textstyle', 'big', 'Big', 'bigl', 'bigr'}
UNICODE_MAP = {'−': '-', '×': '*', '·': '*', '÷': '/', '√': ' sqrt ', 'π': ' pi ', '²': '**2',
               '³': '**3', '⁴': '**4', '⁵': '**5', '（': '(', '）': ')', '∙': '*', '⋅': '*'}


def grab_group(s, i):
    """return (content, index after it). s[i] may be '{' (nested ok) or a single token."""
    while i < len(s) and s[i] == ' ':
        i += 1
    if i >= len(s):
        return '', i
    if s[i] == '{':
        depth, j = 0, i
        while j < len(s):
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)          # unbalanced, take the rest
    m = re.match(r'\\[A-Za-z]+|\d+(\.\d+)?|[A-Za-z]|.', s[i:])
    return m.group(0), i + m.end()


def expand_latex(s, notes):
    """turn \\frac{a}{b}, \\sqrt{x}, \\sin ... into plain text"""
    out, i = '', 0
    while i < len(s):
        if s[i] != '\\':
            out += s[i]
            i += 1
            continue
        m = re.match(r'\\([A-Za-z]+|.)', s[i:])
        name = m.group(1)
        i += m.end()
        if name in ('frac', 'dfrac', 'tfrac'):
            a, i = grab_group(s, i)
            b, i = grab_group(s, i)
            out += '((' + expand_latex(a, notes) + ')/(' + expand_latex(b, notes) + '))'
        elif name == 'sqrt':
            n = None
            if i < len(s) and s[i] == '[':
                j = s.index(']', i)
                n, i = s[i + 1:j], j + 1
            a, i = grab_group(s, i)
            if n:
                out += '((' + expand_latex(a, notes) + ')**(1/(' + expand_latex(n, notes) + ')))'
            else:
                out += 'sqrt(' + expand_latex(a, notes) + ')'
        elif name in LATEX_WORD:
            out += name + ' '
        elif name in LATEX_MAP:
            out += LATEX_MAP[name]
        elif name in LATEX_DROP:
            out += ' '
        elif name in ('lvert', 'rvert', 'vert'):
            out += '|'
        else:
            notes.append("dropped unknown LaTeX command \\%s" % name)
            out += ' '
    return out


def normalize(text, notes):
    """unicode + LaTeX + ^ + braces + |x|  ->  something Python's tokenizer accepts"""
    s = text.strip().strip('$')
    for k, v in UNICODE_MAP.items():
        s = s.replace(k, v)
    if '\\' in s:
        s = expand_latex(s, notes)
        notes.append("LaTeX read as plain math")
    s = re.sub(r'_\{(\w+)\}', r'_\1', s)           # x_{12} -> x_12
    s = s.replace('^', '**').replace('{', '(').replace('}', ')')
    if '|' in s:                                     # |x| -> abs(x), not nested
        parts, opened, res = s.split('|'), False, ''
        for k, p in enumerate(parts):
            res += p
            if k < len(parts) - 1:
                res += ')' if opened else 'abs('
                opened = not opened
        s = res
        notes.append("|...| read as abs(...)")
    # 2x -> 2 x  (but keep 1e-3)
    s = re.sub(r'(?<![A-Za-z_])(\d+\.?\d*)(?![eE][+-]?\d)(?=[A-Za-z(])', r'\1 ', s)
    return s


def smart_tokens(src, declared, notes):
    """implicit multiplication, split 'ac' into a*c, 'sin x' -> sin(x), aliases"""
    raw = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for k, t in enumerate(raw):                      # 'foo(' with unknown foo is a function typo
        if t.type == tokenize.NAME and len(t.string) > 1 and ALIASES.get(t.string, t.string) not in FUNCS \
                and t.string not in declared and k + 1 < len(raw) and raw[k + 1].string == '(':
            raise ValueError("unknown function %s. I know %s" % (t.string, ', '.join(sorted(FUNCS | set(ALIASES)))))
    toks = []
    for t in raw:
        if t.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER,
                      tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
            continue
        if t.type == tokenize.ERRORTOKEN and t.string.strip() == '':
            continue
        if t.type == tokenize.NAME:
            w = ALIASES.get(t.string, t.string)
            if w != t.string:
                notes.append("%s read as %s" % (t.string, w))
            keep = (w in FUNCS or w in CONSTS or w in KEEP or w in declared
                    or re.fullmatch(r'[A-Za-z](_?\d+)?', w))
            if keep:
                toks.append(('NAME', w))
            else:
                parts = re.findall(r'[A-Za-z](?:_?\d+)?', w)
                notes.append("'%s' split into %s (no variable called %s)"
                             % (w, '*'.join(parts), w))
                for p in parts:
                    toks.append(('NAME', p))
        elif t.type == tokenize.NUMBER:
            toks.append(('NUM', t.string))
        elif t.type == tokenize.OP:
            toks.append(('OP', t.string))
        else:
            raise ValueError("cannot read %r" % t.string)
    # sin x  ->  sin ( x )
    out, k = [], 0
    while k < len(toks):
        kind, s = toks[k]
        out.append(toks[k])
        if kind == 'NAME' and s in FUNCS and k + 1 < len(toks) and toks[k + 1] != ('OP', '('):
            nxt = toks[k + 1]
            if nxt[0] in ('NAME', 'NUM'):
                out += [('OP', '('), nxt, ('OP', ')')]
                notes.append("%s %s read as %s(%s)" % (s, nxt[1], s, nxt[1]))
                k += 1
        k += 1
    toks = out
    # implicit multiplication
    def is_value_end(tok):
        return (tok[0] in ('NAME', 'NUM') and tok[1] not in FUNCS) or tok == ('OP', ')')

    def is_value_start(tok):
        return tok[0] in ('NAME', 'NUM') or tok == ('OP', '(')

    res = []
    for tok in toks:
        if res and is_value_end(res[-1]) and is_value_start(tok):
            if not (res[-1][0] == 'NAME' and res[-1][1] in FUNCS):
                res.append(('OP', '*'))
                notes.append("inserted * between %s and %s" % (res[-2][1], tok[1]))
        res.append(tok)
    return ' '.join(s for _, s in res)


def read_formula(text, declared=()):
    """
    text may be one expression, or several lines  't1 = b*b' ... ending in the output.
    returns dict(script=[(name, pysrc), ...], out=name_or_None, src=final pysrc,
                 vars=[...], notes=[...])
    """
    notes = []
    declared = set(declared)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    script = []
    assigned = []
    for ln in lines:
        name = None
        if '=' in ln and not re.search(r'[<>!=]=', ln):
            lhs, rhs = ln.split('=', 1)
            name = lhs.strip()
            if not re.fullmatch(r'[A-Za-z_]\w*', name):
                raise ValueError("bad name on the left of '=' in %r" % ln)
            ln = rhs
        py = smart_tokens(normalize(ln, notes), declared | set(assigned), notes)
        py = ast.unparse(ast.parse(py, mode='eval'))   # syntax check + tidy spacing
        script.append((name, py))
        if name:
            assigned.append(name)
    # variables = names used that are never assigned, in order of first use
    seen, variables = set(), []
    for _, py in script:
        names = [nd for nd in ast.walk(ast.parse(py, mode='eval')) if isinstance(nd, ast.Name)]
        for node in sorted(names, key=lambda nd: (nd.lineno, nd.col_offset)):    # order of appearance
            if True:
                n = node.id
                if n in FUNCS or n in CONSTS or n in assigned or n in seen:
                    continue
                seen.add(n)
                variables.append(n)
    notes = list(dict.fromkeys(notes))                 # unique, keep order
    return dict(script=script, src=script[-1][1], out=script[-1][0],
                vars=variables, notes=notes)


# ----------------------------------------------------------------------------
# 2.  THE TAPE   (python AST -> one line per rounded operation)
# ----------------------------------------------------------------------------
@dataclass
class Line:
    idx: int
    op: str                     # input const add sub mul div neg sq sqrt exp ...
    args: list = field(default_factory=list)   # indices of operand lines
    name: str = ''              # input name or const literal
    text: str = ''              # pretty form, e.g.  t3 = sqrt(t2)
    node: object = None         # the AST node that made this line
    # filled by the forward pass
    value: float = 0.0          # computed (rounded) value  t_hat_i
    exact: object = None        # exact op on the rounded operands (Fraction or mpf)
    slip: float = 0.0           # rho_i = t_hat_i - exact   (Linnainmaa's r_i)
    delta: float = 0.0          # relative slip rho_i / exact
    d: float = 1.0              # a priori error multiplier (0 exact, 1 IEEE op, 2 libm)
    flags: list = field(default_factory=list)
    partials: list = field(default_factory=list)   # local derivatives wrt args
    # filled by the backward pass
    lam: float = 0.0            # lambda_i = dy / dt_i
    # diagnosis helpers
    digits_lost: float = 0.0    # for add/sub lines
    absorbed: bool = False      # tiny operand swallowed by a big one


class Tape:
    def __init__(self):
        self.lines = []
        self.inputs = {}        # name -> idx
        self.env = {}           # assigned names -> idx
        self.node_line = {}     # id(ast node) -> idx
        self.out = None

    def label(self, idx):
        ln = self.lines[idx]
        if ln.op == 'input':
            return ln.name
        if ln.op == 'const':
            return ln.name
        return 't%d' % ln.idx

    def add(self, op, args=(), name='', node=None):
        idx = len(self.lines)
        ln = Line(idx, op, list(args), name=name, node=node)
        self.lines.append(ln)
        if node is not None:
            self.node_line[id(node)] = idx
        return idx

    def finish_text(self):
        tn = 0
        for ln in self.lines:
            if ln.op in ('input', 'const'):
                ln.text = '%s (%s)' % (self.label(ln.idx), ln.op)
                continue
            a = [self.label(j) for j in ln.args]
            sym = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
            if ln.op in sym:
                rhs = '%s %s %s' % (a[0], sym[ln.op], a[1])
            elif ln.op == 'neg':
                rhs = '-%s' % a[0]
            elif ln.op == 'sq':
                rhs = '%s * %s' % (a[0], a[0])
            else:
                rhs = '%s(%s)' % (ln.op, ', '.join(a))
            ln.text = 't%d = %s' % (ln.idx, rhs)

    # ---- building from source -------------------------------------------
    @staticmethod
    def from_reading(reading):
        tape = Tape()
        for v in reading['vars']:
            tape.inputs[v] = tape.add('input', name=v)
        for name, py in reading['script']:
            tree = ast.parse(py, mode='eval')
            idx = tape.build(tree.body, py)
            if name:
                tape.env[name] = idx
            tape.out = idx
        tape.finish_text()
        return tape

    def build(self, node, src):
        if isinstance(node, ast.Constant):
            lit = ast.get_source_segment(src, node) or repr(node.value)
            return self.add('const', name=lit, node=node)
        if isinstance(node, ast.Name):
            n = node.id
            if n in self.env:
                return self.env[n]
            if n in self.inputs:
                return self.inputs[n]
            if n == 'pi':
                return self.add('const', name='pi', node=node)
            if n == 'e':
                return self.add('const', name='e', node=node)
            raise ValueError("unknown name %s" % n)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                lit = ast.get_source_segment(src, node.operand) or repr(node.operand.value)
                return self.add('const', name='-' + lit, node=node)
            a = self.build(node.operand, src)
            if isinstance(node.op, ast.USub):
                return self.add('neg', [a], node=node)
            if isinstance(node.op, ast.UAdd):
                return a
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return self.build_pow(node, src)
            a = self.build(node.left, src)
            b = self.build(node.right, src)
            ops = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}
            for k, v in ops.items():
                if isinstance(node.op, k):
                    return self.add(v, [a, b], node=node)
            raise ValueError("unsupported operator")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f = node.func.id
            if f not in FUNCS:
                raise ValueError("unknown function %s" % f)
            args = [self.build(x, src) for x in node.args]
            if f == 'pow':
                return self.build_pow(ast.BinOp(node.args[0], ast.Pow(), node.args[1]), src,
                                      prebuilt=args, node_for=node)
            if len(args) != 1:
                raise ValueError("%s takes one argument" % f)
            return self.add(f, args, node=node)
        raise ValueError("unsupported syntax: %s" % ast.dump(node))

    def build_pow(self, node, src, prebuilt=None, node_for=None):
        """x**2 -> one multiply, x**n -> chain of multiplies, x**0.5 -> sqrt, else libm pow"""
        key = node if node_for is None else node_for
        if (not prebuilt and isinstance(node.left, ast.Name) and node.left.id == 'e'
                and 'e' not in self.inputs and 'e' not in self.env):
            return self.add('exp', [self.build(node.right, src)], node=key)   # e**x is exp(x)
        base = prebuilt[0] if prebuilt else self.build(node.left, src)
        exp = node.right
        if isinstance(exp, ast.UnaryOp) and isinstance(exp.op, ast.USub) \
                and isinstance(exp.operand, ast.Constant):
            exp = ast.Constant(-exp.operand.value)
        if isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)):
            n = exp.value
            if n == 2:
                return self.add('sq', [base], node=key)
            if n == 0.5:
                return self.add('sqrt', [base], node=key)
            if n == 1:
                return base
            if n == -1:
                one = self.add('const', name='1')
                return self.add('div', [one, base], node=key)
            if float(n).is_integer() and 3 <= n <= 8:
                cur = self.add('sq', [base])
                for _ in range(int(n) - 2):
                    cur = self.add('mul', [cur, base])
                self.node_line[id(key)] = cur
                return cur
        e = prebuilt[1] if prebuilt else self.build(node.right, src)
        return self.add('pow', [base, e], node=key)


# ----------------------------------------------------------------------------
# 3.  FORWARD PASS   (values, exact local results, slips, exactness rules)
# ----------------------------------------------------------------------------
def round_to_bits(fr, p):
    """round an exact Fraction to the nearest number with p significant bits (ties to even)"""
    fr = Fraction(fr)
    if fr == 0:
        return 0.0
    if p >= 53:
        return float(fr)                      # float() is correctly rounded
    sign = -1 if fr < 0 else 1
    a = abs(fr)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    k = e - (p - 1)                           # scaled = a / 2^k lies in [2^(p-1), 2^p)
    scaled = a / (Fraction(2) ** k)
    q = round(scaled)                         # Fraction.__round__ is round half to even
    if q == 2 ** p:
        q, k = 2 ** (p - 1), k + 1
    return sign * math.ldexp(q, k)


def is_pow2(x):
    return x != 0 and math.isfinite(x) and math.frexp(abs(x))[0] == 0.5


def native(op, a):
    """the real machine, binary64 ops and the C library functions"""
    f = {'add': lambda: a[0] + a[1], 'sub': lambda: a[0] - a[1],
         'mul': lambda: a[0] * a[1], 'div': lambda: a[0] / a[1],
         'neg': lambda: -a[0], 'sq': lambda: a[0] * a[0], 'abs': lambda: abs(a[0]),
         'sqrt': lambda: math.sqrt(a[0]), 'exp': lambda: math.exp(a[0]),
         'log': lambda: math.log(a[0]), 'log10': lambda: math.log10(a[0]),
         'log2': lambda: math.log2(a[0]), 'sin': lambda: math.sin(a[0]),
         'cos': lambda: math.cos(a[0]), 'tan': lambda: math.tan(a[0]),
         'asin': lambda: math.asin(a[0]), 'acos': lambda: math.acos(a[0]),
         'atan': lambda: math.atan(a[0]), 'sinh': lambda: math.sinh(a[0]),
         'cosh': lambda: math.cosh(a[0]), 'tanh': lambda: math.tanh(a[0]),
         'expm1': lambda: math.expm1(a[0]), 'log1p': lambda: math.log1p(a[0]),
         'pow': lambda: math.pow(a[0], a[1])}
    return f[op]()


def exact_local(op, a):
    """the SAME operation done exactly on the rounded operands (Linnainmaa eq. 3)"""
    if op in ('add', 'sub', 'mul', 'div', 'neg', 'sq', 'abs'):
        fa = [Fraction(x) for x in a]
        if op == 'add':
            return fa[0] + fa[1]
        if op == 'sub':
            return fa[0] - fa[1]
        if op == 'mul':
            return fa[0] * fa[1]
        if op == 'div':
            return fa[0] / fa[1]
        if op == 'neg':
            return -fa[0]
        if op == 'sq':
            return fa[0] * fa[0]
        return abs(fa[0])
    ha = [hp(x) for x in a]
    return hp_func(op, *ha)


def partials(op, a, t):
    """local derivatives d(line)/d(operand) at the rounded values"""
    if op == 'add':
        return [1.0, 1.0]
    if op == 'sub':
        return [1.0, -1.0]
    if op == 'mul':
        return [a[1], a[0]]
    if op == 'div':
        return [1.0 / a[1], -a[0] / (a[1] * a[1])]
    if op == 'neg':
        return [-1.0]
    if op == 'sq':
        return [2.0 * a[0]]
    if op == 'abs':
        return [math.copysign(1.0, a[0]) if a[0] != 0 else 0.0]
    if op == 'sqrt':
        return [1.0 / (2.0 * t) if t != 0 else math.inf]
    if op == 'exp':
        return [t]
    if op == 'log':
        return [1.0 / a[0]]
    if op == 'log10':
        return [1.0 / (a[0] * math.log(10))]
    if op == 'log2':
        return [1.0 / (a[0] * math.log(2))]
    if op == 'sin':
        return [math.cos(a[0])]
    if op == 'cos':
        return [-math.sin(a[0])]
    if op == 'tan':
        return [1.0 + t * t]
    if op == 'asin':
        return [1.0 / math.sqrt(1 - a[0] ** 2)]
    if op == 'acos':
        return [-1.0 / math.sqrt(1 - a[0] ** 2)]
    if op == 'atan':
        return [1.0 / (1 + a[0] ** 2)]
    if op == 'sinh':
        return [math.cosh(a[0])]
    if op == 'cosh':
        return [math.sinh(a[0])]
    if op == 'tanh':
        return [1.0 - t * t]
    if op == 'expm1':
        return [t + 1.0]
    if op == 'log1p':
        return [1.0 / (1.0 + a[0])]
    if op == 'pow':
        x, y = a
        dx = y * math.pow(x, y - 1) if x != 0 else 0.0
        dy = t * math.log(x) if x > 0 else math.nan
        return [dx, dy]
    raise ValueError(op)


def small_integers(a, bits):
    """all operands integer valued and small enough that + - * stay exact"""
    return all(float(x).is_integer() and abs(x) < 2.0 ** (bits // 2) for x in a)


def exactness(op, a, bits):
    """a priori multiplier d_i and the rule that justifies it (Sterbenz, powers of two ...)"""
    if op in ('neg', 'abs'):
        return 0.0, ['exact (sign only)']
    if op in ('add', 'sub', 'mul', 'sq') and small_integers(a, bits):
        return 0.0, ['exact (small integers)']
    if op in ('add', 'sub'):
        x, y = a[0], (-a[1] if op == 'sub' else a[1])
        if x == 0 or y == 0:
            return 0.0, ['exact (adds zero)']
        if x * y < 0:
            X, Y = abs(x), abs(y)
            if Y / 2 <= X <= 2 * Y:
                return 0.0, ['exact (Sterbenz)']
        return 1.0, []
    if op == 'mul':
        if is_pow2(a[0]) or is_pow2(a[1]):
            return 0.0, ['exact (times a power of two)']
        return 1.0, []
    if op == 'div':
        if is_pow2(a[1]):
            return 0.0, ['exact (divide by a power of two)']
        return 1.0, []
    if op == 'sq':
        if is_pow2(a[0]):
            return 0.0, ['exact (power of two squared)']
        return 1.0, []
    if op == 'sqrt':
        return 1.0, []                          # IEEE: correctly rounded
    if bits == 53:
        return 2.0, ['library function, allowed 1 ulp = 2u']
    return 1.0, []                              # emulated precision is correctly rounded


def forward(tape, values, bits):
    """values: name -> Fraction (exact typed number).  Fills value/exact/slip/d/partials."""
    for ln in tape.lines:
        if ln.op == 'input':
            fr = Fraction(values[ln.name])
            ln.exact = fr
            ln.value = round_to_bits(fr, bits)
            ln.slip = float(Fraction(ln.value) - fr)
            ln.delta = ln.slip / float(fr) if fr != 0 else 0.0
            ln.d = 0.0
            ln.flags = [] if ln.slip == 0 else ['typed value was rounded when stored']
            continue
        if ln.op == 'const':
            if ln.name == 'pi':
                fr = hp_to_fraction(mpmath.pi + 0) if HAVE_MP else Fraction(math.pi)
            elif ln.name == 'e':
                fr = hp_to_fraction(mpmath.e + 0) if HAVE_MP else Fraction(math.e)
            else:
                fr = Fraction(ln.name)
            ln.exact = fr
            ln.value = round_to_bits(fr, bits)
            ln.slip = float(Fraction(ln.value) - fr)
            ln.delta = ln.slip / float(fr) if fr != 0 else 0.0
            ln.d = 0.0 if ln.slip == 0 else 1.0
            ln.flags = ['exact constant'] if ln.slip == 0 else ['constant rounded when stored']
            continue
        a = [tape.lines[j].value for j in ln.args]
        try:
            ln.exact = exact_local(ln.op, a)
            if bits == 53:
                ln.value = native(ln.op, a)
            else:
                ln.value = round_to_bits(hp_to_fraction(ln.exact), bits)
            if isinstance(ln.exact, Fraction):
                slip = Fraction(ln.value) - ln.exact
                ln.delta = float(slip / ln.exact) if ln.exact != 0 else 0.0
                ln.slip = float(slip)
            else:
                slip = hp(ln.value) - ln.exact
                ln.delta = hp_float(slip / ln.exact) if ln.exact != 0 else 0.0
                ln.slip = hp_float(slip)
        except (ZeroDivisionError, ValueError, OverflowError, TypeError, AttributeError) as err:
            raise ValueError("line %s failed (%s)" % (ln.text or ln.idx, err))
        if not math.isfinite(ln.value):
            raise ValueError("line %s is not finite (%r)" % (ln.text, ln.value))
        ln.d, ln.flags = exactness(ln.op, a, bits)
        if ln.slip == 0 and ln.d != 0:
            ln.flags.append('measured exact')
        ln.partials = partials(ln.op, a, ln.value)
        if ln.op in ('add', 'sub'):
            big = max(abs(a[0]), abs(a[1]))
            small = min(abs(a[0]), abs(a[1]))
            if big > 0 and ln.value != 0:
                ln.digits_lost = max(0.0, math.log10(big / abs(ln.value)))
            elif big > 0:
                ln.digits_lost = math.inf
            if big > 0 and small > 0 and small / big < 1e-3 and ln.digits_lost < 0.5:
                ln.absorbed = True


def reference(tape):
    """the whole formula in 60 digits, from the STORED inputs and the EXACT constants"""
    vals = []
    for ln in tape.lines:
        if ln.op == 'input':
            vals.append(hp(ln.value))
        elif ln.op == 'const':
            vals.append(hp(ln.exact))
        else:
            a = [vals[j] for j in ln.args]
            if ln.op in ('add', 'sub', 'mul', 'div', 'neg', 'sq', 'abs'):
                r = {'add': lambda: a[0] + a[1], 'sub': lambda: a[0] - a[1],
                     'mul': lambda: a[0] * a[1], 'div': lambda: a[0] / a[1],
                     'neg': lambda: -a[0], 'sq': lambda: a[0] * a[0],
                     'abs': lambda: abs(a[0])}[ln.op]()
            else:
                r = hp_func(ln.op, *a)
            vals.append(r)
    return vals[tape.out]


# ----------------------------------------------------------------------------
# 4.  BACKWARD PASS   (Linnainmaa's Algorithm T  =  solve F_t^T lambda = w)
# ----------------------------------------------------------------------------
def backward(tape):
    for ln in tape.lines:
        ln.lam = 0.0
    tape.lines[tape.out].lam = 1.0                       # T1  start with 1 at the output
    for ln in reversed(tape.lines):                      # T6  walk the tape backwards
        if ln.op in ('input', 'const'):
            continue
        for j, dj in zip(ln.args, ln.partials):          # T5  pass the blame to each operand
            tape.lines[j].lam += ln.lam * dj             #     (+= handles a value used twice)


# ----------------------------------------------------------------------------
# 5.  THE NUMBERS
# ----------------------------------------------------------------------------
@dataclass
class Result:
    src: str
    reading: dict
    tape: Tape
    bits: int
    u: float
    values: dict
    y: float = 0.0
    y_true: object = None
    actual: float = 0.0
    predicted: float = 0.0
    ceiling: float = 0.0
    typical: float = 0.0
    ceiling_all1: float = 0.0
    grad: dict = field(default_factory=dict)
    kappa: float = 0.0
    kappa_norm: float = 0.0
    amp: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.0
    input_effect: float = 0.0
    verdict: str = ''
    warnings: list = field(default_factory=list)
    diagnosis: list = field(default_factory=list)
    candidates: list = field(default_factory=list)      # (pysrc, why)
    label: str = ''


def analyze(reading, values, bits=53, label=''):
    tape = Tape.from_reading(reading)
    forward(tape, values, bits)
    backward(tape)
    R = Result(src=reading['src'], reading=reading, tape=tape, bits=bits,
               u=2.0 ** (-bits), values=values, label=label)
    out = tape.lines[tape.out]
    R.y = out.value
    R.y_true = reference(tape)
    R.actual = hp_float(hp(R.y) - R.y_true)
    comp = [ln for ln in tape.lines if ln.op not in ('input',)]
    R.predicted = sum(ln.lam * ln.slip for ln in comp)
    v = [ln.lam * ln.d * ln.value for ln in comp]
    R.ceiling = R.u * sum(abs(x) for x in v)
    R.typical = R.u * math.sqrt(sum(x * x for x in v)) / math.sqrt(3)
    R.ceiling_all1 = R.u * sum(abs(ln.lam * ln.value) for ln in comp
                               if ln.op not in ('neg', 'abs') and (ln.op != 'const' or ln.d != 0))
    for name, idx in tape.inputs.items():
        R.grad[name] = tape.lines[idx].lam
    ay = abs(R.y)
    xs = [tape.lines[i].value for i in tape.inputs.values()]
    gs = [R.grad[n] for n in tape.inputs]
    gx = sum(abs(g * x) for g, x in zip(gs, xs))
    gnorm = math.sqrt(sum(g * g for g in gs))
    xnorm = math.sqrt(sum(x * x for x in xs))
    R.kappa = gx / ay if ay > 0 else math.inf
    R.kappa_norm = gnorm * xnorm / ay if ay > 0 else math.inf
    R.amp = (R.ceiling / R.u) / ay if ay > 0 else math.inf
    if gnorm * xnorm > 0:
        R.beta1 = sum(abs(x) for x in v) / (gnorm * xnorm)
        R.beta2 = math.sqrt(sum(x * x for x in v)) / (gnorm * xnorm)
    R.input_effect = sum(R.grad[n] * tape.lines[i].slip for n, i in tape.inputs.items())
    # verdict: compare what the algorithm allows (A) with what the problem forces (kappa).
    # The allowance grows with the number of rounded steps, like the n*eps bounds Bean proves
    # for sums and Horner. A fixed 10 called a 49 step sum unstable, which it is not.
    R.steps = sum(1 for ln in comp if ln.op != 'const')
    R.allow = max(10.0, float(R.steps))
    if ay == 0:
        R.verdict = 'output is 0, relative measures undefined'
    elif R.amp <= R.allow * max(1.0, R.kappa):
        R.verdict = 'STABLE  (the error ceiling is at the level the problem itself forces)'
    else:
        R.verdict = 'UNSTABLE  (the algorithm allows %.1e times more error than the problem forces)' \
                    % (R.amp / max(1.0, R.kappa))
    if ay > 0 and R.ceiling / ay > 0.1:
        R.warnings.append('ceiling is above 10% of |y|, first order analysis is not reliable here')
    if abs(R.actual) > 0.01 * R.ceiling and R.ceiling > 0:
        mism = abs(R.predicted - R.actual) / abs(R.actual)
        if mism > 0.1:
            R.warnings.append('predicted and actual error differ by %.0f%%, second order effects present'
                              % (100 * mism))
    diagnose(R)
    return R


# ----------------------------------------------------------------------------
# 6.  WHY  (rule based explanation)  and  WHAT TO DO  (rewrite candidates)
# ----------------------------------------------------------------------------
def consumers_of(tape):
    cons = {ln.idx: [] for ln in tape.lines}
    for ln in tape.lines:
        for j in ln.args:
            cons[j].append(ln.idx)
    return cons


def worst_cancellation_downstream(tape, idx, cons):
    """the cancelling add/sub line reachable from idx that loses the most digits"""
    best, stack, seen = None, list(cons[idx]), set()
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        ln = tape.lines[k]
        if ln.op in ('add', 'sub') and ln.digits_lost >= 1:
            if best is None or ln.digits_lost > best.digits_lost:
                best = ln
        stack += cons[k]
    return best


def diagnose(R):
    tape, ay = R.tape, abs(R.y)
    comp = [ln for ln in tape.lines if ln.op != 'input']
    total = sum(abs(ln.lam * ln.d * ln.value) for ln in comp)
    cons = consumers_of(tape)
    D = R.diagnosis
    if total == 0:
        D.append('every line is exact at these inputs, the computed value has no rounding error')
        return
    ranked = sorted(comp, key=lambda ln: -abs(ln.lam * ln.d * ln.value))
    for ln in ranked[:4]:
        share = abs(ln.lam * ln.d * ln.value) / total
        if share < 0.03:
            break
        amp = abs(ln.lam * ln.value) / ay if ay > 0 else math.inf
        s = "line %s carries %.0f%% of the ceiling (weight |lambda t| / |y| = %s, so one u of slip here becomes %s u in y)." \
            % (tape.label(ln.idx), 100 * share, fnum(amp), fnum(amp))
        c = worst_cancellation_downstream(tape, ln.idx, cons)
        if c is not None:
            s += " Its result feeds the cancelling %s at %s, which throws away %.1f digits and keeps the slip." \
                 % ('subtraction' if c.op == 'sub' else 'addition of opposite signs', tape.label(c.idx), c.digits_lost)
        elif ln.op in ('add', 'sub') and ln.absorbed and 'measured exact' not in ln.flags:
            s += " This step absorbs a tiny operand into a big one, so the tiny operand's digits are lost."
        elif ln.d == 2:
            s += " It is a library function (allowed 1 ulp)."
        elif ln.op == 'const':
            s += " The constant %s is not a binary fraction, so storing it already rounds." % ln.name
        else:
            s += " No digit loss here, plain accumulation."
        D.append(s)
    exact_lines = [tape.label(ln.idx) for ln in comp if ln.d == 0 and ln.op != 'const']
    if exact_lines:
        D.append('exact lines (add no slip of their own): ' + ', '.join(exact_lines))
    meas = [tape.label(ln.idx) for ln in comp if 'measured exact' in ln.flags]
    if meas:
        D.append('measured exact at these inputs although the model allows a slip: ' + ', '.join(meas))
    if R.kappa > 1e3:
        D.append('the problem itself is ill conditioned here (kappa = %s), any algorithm must lose about log10(kappa) = %.1f digits'
                 % (fnum(R.kappa), math.log10(R.kappa)))
    # rewrite candidates, only when a rewrite could help
    if ay > 0 and (R.amp > R.allow * max(1.0, R.kappa) or R.amp > 100) and len(R.reading['script']) == 1:
        R.candidates = find_rewrites(R)
        if not R.candidates:
            cancel = [ln for ln in comp if ln.op in ('add', 'sub') and ln.digits_lost >= 1]
            if cancel:
                D.append('no automatic rewrite matched the cancellation at %s. By hand: conjugate trick, a series for small inputs, or do that line in higher precision.'
                         % ', '.join(tape.label(c.idx) for c in cancel))


# ---- rewrite patterns (AST level) --------------------------------------------
def same(a, b):
    return ast.dump(a) == ast.dump(b)


def is_call(n, f):
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == f and len(n.args) == 1


def exp_arg(n):
    """returns X if n is exp(X) or e**X"""
    if is_call(n, 'exp'):
        return n.args[0]
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and isinstance(n.left, ast.Name) and n.left.id == 'e':
        return n.right
    return None


def is_const(n, val):
    return isinstance(n, ast.Constant) and n.value == val


def is_square_of(n):
    """returns A if n is A**2 or A*A"""
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and is_const(n.right, 2):
        return n.left
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and same(n.left, n.right):
        return n.left
    return None


def neg(n):
    return ast.UnaryOp(ast.USub(), n)


def call(f, *a):
    return ast.Call(ast.Name(f, ast.Load()), list(a), [])


def binop(a, op, b):
    return ast.BinOp(a, op, b)


def rewrite_node(n):
    """returns (new_node, description) or None"""
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
        L, Rr = n.left, n.right
        # forms with a square root:  -A + sqrt(A^2 - R),  sqrt(A^2 - R) - A,  A - sqrt(A^2 - R), sqrt(A^2 + R) - A
        if isinstance(n.op, ast.Add) and isinstance(L, ast.UnaryOp) and isinstance(L.op, ast.USub):
            A, S = L.operand, Rr
            sign = +1
        elif isinstance(n.op, ast.Sub) and is_call(L, 'sqrt'):
            A, S, sign = Rr, L, +1          # sqrt(..) - A
        elif isinstance(n.op, ast.Sub) and is_call(Rr, 'sqrt'):
            A, S, sign = L, Rr, -1          # A - sqrt(..)
        else:
            A = S = None
        if A is not None and is_call(S, 'sqrt'):
            inner = S.args[0]
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, (ast.Sub, ast.Add)):
                sq = is_square_of(inner.left)
                if sq is not None and same(sq, A):
                    Rpart = inner.right
                    if isinstance(inner.op, ast.Sub):       # A^2 - R
                        if sign == +1:   # sqrt(A^2-R) - A  = -R / (A + sqrt(A^2-R))
                            new = binop(neg(Rpart), ast.Div(), binop(A, ast.Add(), S))
                            return new, 'conjugate trick, sqrt(A^2 - R) - A = -R/(A + sqrt(A^2 - R))', r'\sqrt{A^2-R}-A=\frac{-R}{A+\sqrt{A^2-R}}'
                        else:            # A - sqrt(A^2-R)  =  R / (A + sqrt(A^2-R))
                            new = binop(Rpart, ast.Div(), binop(A, ast.Add(), S))
                            return new, 'conjugate trick, A - sqrt(A^2 - R) = R/(A + sqrt(A^2 - R))', r'A-\sqrt{A^2-R}=\frac{R}{A+\sqrt{A^2-R}}'
                    else:                                   # A^2 + R
                        if sign == +1:   # sqrt(A^2+R) - A = R/(sqrt(A^2+R) + A)
                            new = binop(Rpart, ast.Div(), binop(S, ast.Add(), A))
                            return new, 'conjugate trick, sqrt(A^2 + R) - A = R/(sqrt(A^2 + R) + A)', r'\sqrt{A^2+R}-A=\frac{R}{\sqrt{A^2+R}+A}'
                        else:            # A - sqrt(A^2+R) = -R/(A + sqrt(A^2+R))
                            new = binop(neg(Rpart), ast.Div(), binop(A, ast.Add(), S))
                            return new, 'conjugate trick, A - sqrt(A^2 + R) = -R/(A + sqrt(A^2 + R))', r'A-\sqrt{A^2+R}=\frac{-R}{A+\sqrt{A^2+R}}'
        if isinstance(n.op, ast.Sub):
            if is_const(L, 1) and is_call(Rr, 'cos'):
                X = Rr.args[0]
                half = binop(X, ast.Div(), ast.Constant(2))
                new = binop(ast.Constant(2), ast.Mult(), binop(call('sin', half), ast.Pow(), ast.Constant(2)))
                return new, 'half angle, 1 - cos(X) = 2 sin(X/2)^2', r'1-\cos X=2\sin^2(X/2)'
            if exp_arg(L) is not None and is_const(Rr, 1):
                return call('expm1', exp_arg(L)), 'exp(X) - 1 = expm1(X)', r'e^X-1=\operatorname{expm1}(X)'
            if is_call(L, 'sqrt') and is_call(Rr, 'sqrt'):
                A, B = L.args[0], Rr.args[0]
                new = binop(binop(A, ast.Sub(), B), ast.Div(), binop(L, ast.Add(), Rr))
                return new, 'conjugate trick, sqrt(A) - sqrt(B) = (A - B)/(sqrt(A) + sqrt(B))', r'\sqrt A-\sqrt B=\frac{A-B}{\sqrt A+\sqrt B}'
            sa, sb = is_square_of(L), is_square_of(Rr)
            if sa is not None and sb is not None:
                new = binop(binop(sa, ast.Sub(), sb), ast.Mult(), binop(sa, ast.Add(), sb))
                return new, 'factor, X^2 - Y^2 = (X - Y)(X + Y)', r'X^2-Y^2=(X-Y)(X+Y)'
            if (isinstance(L, ast.BinOp) and isinstance(L.op, ast.Div) and is_const(L.left, 1)
                    and isinstance(Rr, ast.BinOp) and isinstance(Rr.op, ast.Div) and is_const(Rr.left, 1)):
                X, Y = L.right, Rr.right
                new = binop(binop(Y, ast.Sub(), X), ast.Div(), binop(X, ast.Mult(), Y))
                return new, 'combine, 1/X - 1/Y = (Y - X)/(X Y)', r'\frac1X-\frac1Y=\frac{Y-X}{XY}'
    if is_call(n, 'log'):
        inner = n.args[0]
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
            if is_const(inner.left, 1):
                return call('log1p', inner.right), 'log(1 + X) = log1p(X)', r'\ln(1+X)=\operatorname{log1p}(X)'
            if is_const(inner.right, 1):
                return call('log1p', inner.left), 'log(X + 1) = log1p(X)', r'\ln(X+1)=\operatorname{log1p}(X)'
    return None


class Replace(ast.NodeTransformer):
    def __init__(self, target, new):
        self.target, self.new = target, new

    def generic_visit(self, node):
        if node is self.target:
            return self.new
        return super().generic_visit(node)


def find_rewrites(R):
    tree = ast.parse(R.src, mode='eval')
    found = []
    for node in ast.walk(tree):
        rw = rewrite_node(node)
        if rw is None:
            continue
        new_node, why, why_tex = rw
        # work on a fresh parse and find the matching node by its dump (identity differs)
        new_tree = ast.parse(R.src, mode='eval')
        for cand in ast.walk(new_tree):
            if ast.dump(cand) == ast.dump(node):
                new_tree = ast.fix_missing_locations(Replace(cand, new_node).visit(new_tree))
                break
        src = ast.unparse(new_tree)
        if src != R.src and src not in [c[0] for c in found]:
            found.append((src, why, why_tex))
    return found


# ----------------------------------------------------------------------------
# 7.  REPORTS   (text, matrix view, LaTeX)
# ----------------------------------------------------------------------------
def fnum(x):
    if x is None:
        return '?'
    if isinstance(x, str):
        return x
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return '0'
    if 1e-3 <= abs(x) < 1e6:
        s = '%.6g' % x
        return s
    s = '%.3e' % x
    m, e = s.split('e')
    return '%se%d' % (m.rstrip('0').rstrip('.'), int(e))


def flong(x):
    return repr(float(x))


def text_report(R, show_tape=True):
    tape, u, ay = R.tape, R.u, abs(R.y)
    out = []
    w = out.append
    w('=' * 72)
    w('rounderr   %s' % (R.label or ''))
    w('=' * 72)
    w('formula   y = %s' % R.src)
    for n, py in R.reading['script'][:-1]:
        w('          %s = %s' % (n, py))
    w('inputs    ' + '   '.join('%s = %s' % (n, fnum(float(R.values[n]))) for n in tape.inputs))
    w('precision %d bits   u = %s' % (R.bits, fnum(u)))
    if show_tape:
        w('')
        w('TAPE  (one line per rounded step)')
        w('%-24s %-14s %3s %-11s %-11s %5s  %s' % ('line', 'value', 'd', 'lambda', '|lambda d t|', 'share', 'notes'))
        comp = [ln for ln in tape.lines if ln.op != 'input']
        total = sum(abs(ln.lam * ln.d * ln.value) for ln in comp) or 1.0
        for ln in tape.lines:
            if ln.op == 'input':
                w('%-24s %-14s %3s %-11s %-11s %5s  %s' % (ln.text, fnum(ln.value), '', fnum(ln.lam), '', '',
                                                         'gradient dy/d%s' % ln.name))
                continue
            vv = abs(ln.lam * ln.d * ln.value)
            w('%-24s %-14s %3s %-11s %-11s %4.0f%%  %s' % (ln.text, fnum(ln.value), '%g' % ln.d, fnum(ln.lam),
                                                           fnum(vv), 100 * vv / total, ', '.join(ln.flags)))
    w('')
    w('WHAT ACTUALLY HAPPENED')
    w('  computed y    %s' % flong(R.y))
    w('  true y        %s   (60 digits, same stored inputs)' % mpstr(R.y_true))
    rel = R.actual / ay if ay > 0 else math.nan
    w('  actual error  %s   relative %s  =  %s u' % (fnum(R.actual), fnum(rel), fnum(rel / u)))
    w('  predicted     %s   (sum lambda_i rho_i with the measured slips)' % fnum(R.predicted))
    if R.input_effect != 0:
        w('  typed inputs were rounded when stored, that alone moves y by %s' % fnum(R.input_effect))
    w('')
    w('WHAT THE ERROR MIGHT BE  (before knowing the slips)')
    w('  ceiling   %s   =  %s u relative   (worst case, u * sum |lambda d t|)' % (fnum(R.ceiling), fnum(R.amp)))
    w('  typical   %s   (random slips, u * ||lambda d t||_2 / sqrt 3)' % fnum(R.typical))
    if R.ceiling_all1 != R.ceiling:
        w('  ceiling with every line charged once (no exactness credit)   %s  =  %s u relative' % (fnum(R.ceiling_all1), fnum(R.ceiling_all1 / R.u / ay if ay else math.nan)))
    w('  problem conditioning   kappa = %s   (the problem forces about kappa u)' % fnum(R.kappa))
    w('  algorithm ceiling      A = %s u   (A / kappa = %s)' % (fnum(R.amp), fnum(R.amp / max(1.0, R.kappa))))
    w('  backward error (Bindel)  beta1 = %s   beta2 = %s   (in units of u)' % (fnum(R.beta1), fnum(R.beta2)))
    w('  verdict   %s' % R.verdict)
    for s in R.warnings:
        w('  WARNING   %s' % s)
    w('')
    w('WHY')
    for s in R.diagnosis:
        w('  * ' + s)
    if R.candidates:
        w('')
        w('WHAT TO DO  (rewrite candidates, same inputs)')
        for k, c in enumerate(R.candidates, 1):
            w('  [%d] %s      %s' % (k, c[0], c[1]))
    return '\n'.join(out)


def mpstr(v):
    if HAVE_MP:
        return mpmath.nstr(v, 25)
    return str(v)[:30]


def matrix_view(R):
    """the whiteboard picture: L = F_t (unit lower triangular), B = G_x, solve L^T lambda = w"""
    tape = R.tape
    comp = [ln for ln in tape.lines if ln.op != 'input']
    n = len(comp)
    pos = {ln.idx: k for k, ln in enumerate(comp)}
    inputs = list(tape.inputs.items())
    m = len(inputs)
    L = [[0.0] * n for _ in range(n)]
    B = [[0.0] * m for _ in range(n)]
    for k, ln in enumerate(comp):
        L[k][k] = 1.0
        for j, dj in zip(ln.args, ln.partials):
            if j in pos:
                L[k][pos[j]] -= dj
            else:
                col = [i for i, (_, idx) in enumerate(inputs) if idx == j][0]
                B[k][col] += dj
    w = [0.0] * n
    w[pos[tape.out]] = 1.0
    lam = [0.0] * n
    for k in reversed(range(n)):                 # back substitution for L^T lambda = w
        s = w[k] - sum(L[i][k] * lam[i] for i in range(k + 1, n))
        lam[k] = s / L[k][k]
    grad = [sum(B[k][j] * lam[k] for k in range(n)) for j in range(m)]
    lines = ['MATRIX VIEW   F(t,x) = t - G(t,x) = 0,   L = F_t,   B = G_x,   L^T lambda = w,   grad = B^T lambda']
    lines.append('t = (%s)   x = (%s)' % (', '.join(tape.label(ln.idx) for ln in comp), ', '.join(n_ for n_, _ in inputs)))
    if n <= 12:
        lines.append('L =')
        for row in L:
            lines.append('   [' + ' '.join('%10s' % fnum(v) for v in row) + ']')
        lines.append('B =')
        for row in B:
            lines.append('   [' + ' '.join('%10s' % fnum(v) for v in row) + ']')
    lines.append('w      = (%s)' % ', '.join(fnum(v) for v in w))
    lines.append('lambda = (%s)   by back substitution' % ', '.join(fnum(v) for v in lam))
    lines.append('lambda = (%s)   by the sweep' % ', '.join(fnum(ln.lam) for ln in comp))
    lines.append('grad   = (%s)   B^T lambda' % ', '.join(fnum(v) for v in grad))
    err = max(abs(lam[k] - comp[k].lam) for k in range(n)) if n else 0.0
    lines.append('max difference between the two lambdas = %s' % fnum(err))
    R.matrix = dict(L=L, B=B, w=w, lam=lam, grad=grad, tlabels=[tape.label(ln.idx) for ln in comp],
                    xlabels=[n_ for n_, _ in inputs])
    return '\n'.join(lines)


# ---- LaTeX --------------------------------------------------------------------
TEX_HEAD = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\usepackage[margin=0.9in]{geometry}
\definecolor{askblue}{HTML}{1976D2}
\definecolor{amber}{HTML}{FFA000}
\definecolor{trapred}{HTML}{D32F2F}
\definecolor{chipgreen}{HTML}{2E7D32}
\newtcolorbox{rbox}[2][]{breakable,colback=#2!6!white,colframe=#2,fonttitle=\bfseries,#1}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\begin{document}
"""


def tex_num(x):
    if isinstance(x, str):
        return x
    if x is None or not math.isfinite(x):
        return r'\text{%s}' % x
    if x == 0:
        return '0'
    if 1e-3 <= abs(x) < 1e6:
        return '%.6g' % x
    m, e = ('%.3e' % x).split('e')
    return r'%s\times 10^{%d}' % (m.rstrip('0').rstrip('.'), int(e))


def tex_text(s):
    """plain diagnosis sentence -> LaTeX (escape, math for numbers like 2e6)"""
    s = s.replace('\\', r'\textbackslash ')
    for a, b in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('^', r'\^{}'), ('~', r'\~{}')]:
        s = s.replace(a, b)
    s = re.sub(r'(?<![\w.])(-?\d+(?:\.\d+)?)e([+-]?\d+)', lambda m: '$%s\\times 10^{%d}$' % (m.group(1), int(m.group(2))), s)
    s = s.replace('lambda', r'$\lambda$').replace('kappa', r'$\kappa$').replace('|y|', '$|y|$')
    return s


def verb(s):
    s = s.replace('\n', '   ')
    for d in '|!@#+=~':
        if d not in s:
            return r'\verb' + d + s + d
    return r'\texttt{%s}' % tex_text(s)


class TexPrinter:
    """python expression AST -> LaTeX math"""
    PREC = {'add': 1, 'mul': 2, 'neg': 3, 'pow': 4, 'atom': 5}
    FUN = {'sin': r'\sin', 'cos': r'\cos', 'tan': r'\tan', 'exp': r'\exp', 'log': r'\ln',
           'log10': r'\log_{10}', 'log2': r'\log_{2}', 'asin': r'\arcsin', 'acos': r'\arccos',
           'atan': r'\arctan', 'sinh': r'\sinh', 'cosh': r'\cosh', 'tanh': r'\tanh',
           'expm1': r'\operatorname{expm1}', 'log1p': r'\operatorname{log1p}'}

    def __call__(self, src):
        return self.p(ast.parse(src, mode='eval').body)

    def wrap(self, s, need):
        return r'\left(' + s + r'\right)' if need else s

    def p(self, n):
        return self.go(n)[0]

    def go(self, n):
        """returns (latex, precedence)"""
        if isinstance(n, ast.Constant):
            return (tex_num(float(n.value)) if isinstance(n.value, float) else str(n.value)), 5
        if isinstance(n, ast.Name):
            if n.id == 'pi':
                return r'\pi', 5
            m = re.fullmatch(r'([A-Za-z])_?(\d+)', n.id)
            if m:
                return '%s_{%s}' % m.groups(), 5
            return (n.id if len(n.id) == 1 else r'\mathit{%s}' % n.id), 5
        if isinstance(n, ast.UnaryOp):
            s, pr = self.go(n.operand)
            return '-' + self.wrap(s, pr < 3), 3
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Div):
                return r'\frac{%s}{%s}' % (self.p(n.left), self.p(n.right)), 5
            if isinstance(n.op, ast.Pow):
                s, pr = self.go(n.left)
                return '{%s}^{%s}' % (self.wrap(s, pr < 5), self.p(n.right)), 4
            if isinstance(n.op, ast.Mult):
                a, pa = self.go(n.left)
                b, pb = self.go(n.right)
                a = self.wrap(a, pa < 2)
                b = self.wrap(b, pb < 2)
                sep = r' \cdot ' if (b[0].isdigit() or a[-1].isdigit()) else r'\,'
                return a + sep + b, 2
            a, pa = self.go(n.left)
            b, pb = self.go(n.right)
            if isinstance(n.op, ast.Add):
                return self.wrap(a, pa < 1) + ' + ' + self.wrap(b, pb < 1), 1
            return self.wrap(a, pa < 1) + ' - ' + self.wrap(b, pb <= 1), 1
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            f = n.func.id
            if f == 'sqrt':
                return r'\sqrt{%s}' % self.p(n.args[0]), 5
            if f == 'abs':
                return r'\left|%s\right|' % self.p(n.args[0]), 5
            if f == 'pow':
                s, pr = self.go(n.args[0])
                return '{%s}^{%s}' % (self.wrap(s, pr < 5), self.p(n.args[1])), 4
            return r'%s\left(%s\right)' % (self.FUN.get(f, r'\operatorname{%s}' % f), self.p(n.args[0])), 5
        return r'\text{?}', 5


def tex_line(tape, ln):
    """one tape line as LaTeX math"""
    P = TexPrinter()
    def lab(j):
        L = tape.lines[j]
        if L.op == 'input':
            m = re.fullmatch(r'([A-Za-z])_?(\d+)', L.name)
            return '%s_{%s}' % m.groups() if m else (L.name if len(L.name) == 1 else r'\mathit{%s}' % L.name)
        if L.op == 'const':
            return tex_num(float(L.value)) if L.name not in ('pi', 'e') else (r'\pi' if L.name == 'pi' else 'e')
        return 't_{%d}' % L.idx
    a = [lab(j) for j in ln.args]
    if ln.op == 'input':
        return r'%s' % lab(ln.idx)
    if ln.op == 'const':
        return r'%s' % ln.name
    if ln.op in ('add', 'sub', 'mul', 'div'):
        sym = {'add': '+', 'sub': '-', 'mul': r'\cdot', 'div': '/'}[ln.op]
        rhs = '%s %s %s' % (a[0], sym, a[1])
    elif ln.op == 'neg':
        rhs = '-%s' % a[0]
    elif ln.op == 'sq':
        rhs = r'%s \cdot %s' % (a[0], a[0])
    elif ln.op == 'sqrt':
        rhs = r'\sqrt{%s}' % a[0]
    elif ln.op == 'pow':
        rhs = '{%s}^{%s}' % (a[0], a[1])
    else:
        rhs = r'%s(%s)' % (P.FUN.get(ln.op, r'\operatorname{%s}' % ln.op), a[0])
    return 't_{%d} = %s' % (ln.idx, rhs)


def latex_report(results, title='Rounding error report'):
    P = TexPrinter()
    out = [TEX_HEAD, r'\section*{%s}' % tex_text(title)]
    w = out.append
    for R in results:
        tape, u, ay = R.tape, R.u, abs(R.y)
        w(r'\subsection*{%s}' % tex_text(R.label or 'Formula'))
        w(r'\begin{rbox}[title={What is computed}]{askblue}')
        w('As typed ' + verb(R.reading.get('typed', R.src)) + r'\\')
        w('As read ' + verb(R.src) + r'\\')
        w(r'\[ y = %s \]' % P(R.src))
        for n, py in R.reading['script'][:-1]:
            w(r'\[ %s = %s \]' % (n, P(py)))
        w('Inputs ' + ', '.join('$%s = %s$' % (n, tex_num(float(R.values[n]))) for n in tape.inputs)
          + r'.\quad Precision %d bits, $u = %s$.' % (R.bits, tex_num(u)))
        if R.reading['notes']:
            w(r'\\ Reader notes. ' + '. '.join(tex_text(s) for s in R.reading['notes']) + '.')
        w(r'\end{rbox}')
        # tape table
        comp = [ln for ln in tape.lines if ln.op != 'input']
        total = sum(abs(ln.lam * ln.d * ln.value) for ln in comp) or 1.0
        w(r'\subsubsection*{Tape, one line per rounded step}')
        w(r'\begin{longtable}{@{}llrrrrl@{}}')
        w(r'\toprule line & value $t_i$ & $d_i$ & $\lambda_i$ & $|\lambda_i d_i t_i|$ & share & notes \\ \midrule')
        for ln in tape.lines:
            if ln.op == 'input':
                w(r'$%s$ & $%s$ & & $%s$ & & & gradient $\partial y/\partial %s$ \\' %
                  (tex_line(tape, ln), tex_num(ln.value), tex_num(ln.lam), tex_line(tape, ln)))
                continue
            vv = abs(ln.lam * ln.d * ln.value)
            w(r'$%s$ & $%s$ & %g & $%s$ & $%s$ & %.0f\%% & %s \\' %
              (tex_line(tape, ln), tex_num(ln.value), ln.d, tex_num(ln.lam), tex_num(vv),
               100 * vv / total, tex_text(', '.join(ln.flags))))
        w(r'\bottomrule \end{longtable}')
        w(r'\[ \text{ceiling} = u\sum_i|\lambda_i d_i t_i| = %s,\qquad \text{typical} = \frac{u}{\sqrt3}\Big(\sum_i(\lambda_i d_i t_i)^2\Big)^{1/2} = %s \]'
          % (tex_num(R.ceiling), tex_num(R.typical)))
        # numbers
        rel = R.actual / ay if ay > 0 else math.nan
        w(r'\begin{rbox}[title={What actually happened}]{chipgreen}')
        w(r'\begin{tabular}{@{}ll@{}}')
        w(r'computed $\hat y$ & %s \\' % verb(flong(R.y)))
        w(r'true $y$ (60 digits) & %s \\' % verb(mpstr(R.y_true)))
        w(r'actual error $\hat y - y$ & $%s$, relative $%s = %s\,u$ \\' % (tex_num(R.actual), tex_num(rel), tex_num(rel / u)))
        w(r'predicted $\sum_i \lambda_i \rho_i$ (measured slips) & $%s$ \\' % tex_num(R.predicted))
        if R.input_effect != 0:
            w(r'typed inputs rounded when stored, effect on $y$ & $%s$ \\' % tex_num(R.input_effect))
        w(r'\end{tabular}')
        w(r'\end{rbox}')
        w(r'\begin{rbox}[title={What the error might be (before knowing the slips)}]{amber}')
        w(r'\begin{tabular}{@{}ll@{}}')
        w(r'ceiling $u\sum|\lambda_i d_i t_i|$ & $%s$, relative $%s\,u$ \\' % (tex_num(R.ceiling), tex_num(R.amp)))
        w(r'typical (random slips) & $%s$ \\' % tex_num(R.typical))
        if R.ceiling_all1 != R.ceiling:
            w(r'ceiling with every line charged & $%s$ \\' % tex_num(R.ceiling_all1))
        w(r'problem conditioning $\kappa = \sum_j |g_j x_j|/|y|$ & $%s$ \\' % tex_num(R.kappa))
        w(r'algorithm ceiling $A = \sum_i|\lambda_i d_i t_i|/|y|$ & $%s$, so $A/\kappa = %s$ \\' % (tex_num(R.amp), tex_num(R.amp / max(1.0, R.kappa))))
        w(r'backward error $\beta_1 = \|v\|_1/(\|g\|_2\|x\|_2)$, $\beta_2$ & $%s$, $%s$ \\' % (tex_num(R.beta1), tex_num(R.beta2)))
        w(r'\end{tabular}\\[4pt]')
        w(r'\textbf{Verdict} ' + tex_text(R.verdict))
        for s in R.warnings:
            w(r'\\ \textcolor{trapred}{\textbf{Warning}} ' + tex_text(s))
        w(r'\end{rbox}')
        w(r'\begin{rbox}[title={Why}]{trapred}')
        w(r'\begin{itemize}')
        for s in R.diagnosis:
            w(r'\item ' + tex_text(s))
        w(r'\end{itemize}')
        w(r'\end{rbox}')
        if R.candidates:
            w(r'\begin{rbox}[title={What to do, rewrite candidates}]{askblue}')
            w(r'\begin{enumerate}')
            for src, why, why_tex in R.candidates:
                w(r'\item $%s$ \quad {\small $%s$}' % (P(src), why_tex))
            w(r'\end{enumerate}')
            w(r'\end{rbox}')
        if getattr(R, 'matrix', None):
            M = R.matrix
            n = len(M['lam'])
            w(r'\subsubsection*{Matrix view, $F(t,x) = t - G(t,x) = 0$, $L = F_t$, $B = G_x$, $L^{T}\lambda = w$, $\nabla_x y = B^{T}\lambda$}')
            if n <= 8:
                rows = lambda A: r' \\ '.join(' & '.join(tex_num(v) for v in row) for row in A)
                w(r'\[ L = \begin{pmatrix} %s \end{pmatrix},\quad B = \begin{pmatrix} %s \end{pmatrix},\quad w = \begin{pmatrix} %s \end{pmatrix},\quad \lambda = \begin{pmatrix} %s \end{pmatrix} \]'
                  % (rows(M['L']), rows(M['B']), r' \\ '.join(tex_num(v) for v in M['w']), r' \\ '.join(tex_num(v) for v in M['lam'])))
                w(r'\[ \nabla_x y = B^{T}\lambda = (%s) \]' % ', '.join(tex_num(v) for v in M['grad']))
            else:
                w('Tape has %d lines, matrices not printed.' % n)
    if len(results) > 1:
        w(r'\section*{Comparison, same inputs}')
        w(r'\begin{tabular}{@{}lllll@{}} \toprule candidate & $\hat y$ & actual rel. error & ceiling (units of $u$) & verdict \\ \midrule')
        for R in results:
            ay = abs(R.y)
            head = ('$%s$' % P(R.src)) if len(R.reading['script']) == 1 else tex_text(R.label.split('  (')[0])
            w(r'%s & %s & $%s$ & $%s$ & %s \\' % (head, verb(flong(R.y)), tex_num(R.actual / ay if ay else math.nan),
                                                tex_num(R.amp), tex_text(R.verdict.split('  ')[0])))
        w(r'\bottomrule \end{tabular}')
    w(r'\end{document}')
    return '\n'.join(out)


def write_tex(results, path, pdf=False, title='Rounding error report'):
    tex = latex_report(results, title)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    msg = 'wrote %s' % path
    if pdf:
        if shutil.which('pdflatex') is None:
            return msg + '  (pdflatex not found, no PDF)'
        d = os.path.dirname(os.path.abspath(path)) or '.'
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', os.path.basename(path)],
                           cwd=d, capture_output=True, text=True)
        if r.returncode == 0:
            msg += '  and ' + path[:-4] + '.pdf'
        else:
            msg += '  (pdflatex failed, see the .log)'
    return msg


def comparison_text(results):
    out = ['', 'COMPARISON  (same inputs)', '%-24s %-12s %-12s %s' % ('computed y', 'actual rel', 'ceiling (u)', 'verdict')]
    for R in results:
        ay = abs(R.y)
        out.append('  ' + (('y = ' + R.src) if len(R.reading['script']) == 1 else R.label.split('  (')[0]))
        out.append('%-24s %-12s %-12s %s' % (flong(R.y), fnum(R.actual / ay if ay else math.nan),
                                             fnum(R.amp), R.verdict.split('  ')[0]))
    return '\n'.join(out)


# ----------------------------------------------------------------------------
# 8.  optional second opinion from an LLM (off unless --llm)
# ----------------------------------------------------------------------------
def llm_opinion(report_text):
    try:
        import anthropic
    except ImportError:
        return 'pip install anthropic to use --llm'
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return 'set ANTHROPIC_API_KEY to use --llm'
    prompt = ('You are a numerical analyst. Below is a first order rounding error report produced by a '
              'reverse mode tape (Linnainmaa 1976, Bindel 2026). In at most 8 short plain sentences say '
              '(1) what actually happened, (2) what the error might be in other inputs, (3) which line is to blame and why, '
              '(4) the single best fix. No dashes, colons or semicolons.\n\n' + report_text)
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(model=os.environ.get('ROUNDERR_MODEL', 'claude-sonnet-5-5'),
                                     max_tokens=600, messages=[{'role': 'user', 'content': prompt}])
        return msg.content[0].text
    except Exception as err:  # pragma: no cover
        return 'LLM call failed (%s)' % err


# ----------------------------------------------------------------------------
# 9.  examples, self test, command line, interactive loop
# ----------------------------------------------------------------------------
EXAMPLES = [
    ('quadratic root, naive      -b + sqrt(b**2 - c)', '-b + sqrt(b**2 - c)', {'b': '1000', 'c': '1'}),
    ('quadratic root, stable     -c/(b + sqrt(b**2 - c))', '-c/(b + sqrt(b**2 - c))', {'b': '1000', 'c': '1'}),
    ('(1 - cos x)/x**2 at x = 1e-4', '(1 - cos(x))/x**2', {'x': '1e-4'}),
    ('exp(x) - 1 at x = 1e-10', 'exp(x) - 1', {'x': '1e-10'}),
    ('log(1 + x) at x = 1e-10', 'log(1 + x)', {'x': '1e-10'}),
    ('x**2 - y**2 with x close to y', 'x**2 - y**2', {'x': '1.0000001', 'y': '1'}),
    ('sum of four numbers', '0.1 + 0.2 + 0.3 + 0.4', {}),
    ('Horner cubic', '((a*x + b)*x + c)*x + d', {'a': '1', 'b': '-3', 'c': '3', 'd': '-1', 'x': '1.0001'}),
    ('LaTeX input', r'\frac{-b+\sqrt{b^2-4ac}}{2a}', {'a': '1', 'b': '1000', 'c': '1'}),
    ('tape script', 't1 = b*b\nt2 = t1 - c\nt3 = sqrt(t2)\ny = -b + t3', {'b': '1000', 'c': '1'}),
]


def parse_value(text, declared=()):
    """'1e-4', '3/7', 'pi/4', '2**-20' -> exact Fraction (expressions are evaluated in binary64)"""
    text = text.strip()
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        pass
    rd = read_formula(text)
    if rd['vars']:
        raise ValueError('value %r contains unknown names' % text)
    tape = Tape.from_reading(rd)
    forward(tape, {}, 53)
    return Fraction(tape.lines[tape.out].value)


def run(text, values_text, bits=53, label='', declared=None):
    rd = read_formula(text, declared or list(values_text))
    rd['typed'] = text
    vals = {}
    for v in rd['vars']:
        if v not in values_text:
            raise ValueError('no value for %s' % v)
        vals[v] = parse_value(values_text[v])
    return analyze(rd, vals, bits, label)


def selftest():
    print('rounderr self test  (ceilings in units of u, relative)')
    checks = [
        ('a - b exact by Sterbenz', 'a - b', {'a': '1.5', 'b': '1.25'}, lambda R: R.ceiling == 0),
        ('quadratic naive, A about 2e6', '-b + sqrt(b**2 - c)', {'b': '1000', 'c': '1'}, lambda R: 1e6 < R.amp < 4e6 and 'UNSTABLE' in R.verdict),
        ('quadratic stable, A about 2.5', '-c/(b + sqrt(b**2 - c))', {'b': '1000', 'c': '1'}, lambda R: R.amp < 5 and 'STABLE' in R.verdict),
        ('1-cos x over x^2, A about 4e8', '(1 - cos(x))/x**2', {'x': '1e-4'}, lambda R: 1e8 < R.amp < 1e9),
        ('x^2 - y^2 is stable (kappa large)', 'x**2 - y**2', {'x': '1.0000001', 'y': '1'}, lambda R: 'STABLE' in R.verdict and R.kappa > 1e6),
        ('(x-y)(x+y) small ceiling', '(x - y)*(x + y)', {'x': '1.0000001', 'y': '1'}, lambda R: R.amp < 4),
        ('sum, prediction matches actual', '0.1 + 0.2 + 0.3 + 0.4', {}, lambda R: abs(R.predicted - R.actual) <= 1e-3 * abs(R.actual) + 1e-30),
        ('single precision emulation', 'x*y + z', {'x': '1.1', 'y': '2.2', 'z': '3.3'}, lambda R: R.bits == 24 and R.amp < 4),
        ('LaTeX reading', r'\frac{-b+\sqrt{b^2-4ac}}{2a}', {'a': '1', 'b': '1000', 'c': '1'}, lambda R: R.src == '(-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)'),
        ('rewrite found for 1 - cos', '(1 - cos(x))/x**2', {'x': '1e-4'}, lambda R: any('sin' in c[0] for c in R.candidates)),
        ('rewrite found for quadratic', '-b + sqrt(b**2 - c)', {'b': '1000', 'c': '1'}, lambda R: any('-c / (b + sqrt' in c[0] for c in R.candidates)),
    ]
    ok = 0
    for name, f, vals, test in checks:
        bits = 24 if 'single' in name else 53
        try:
            R = run(f, vals, bits, name)
            good = test(R)
            print('  %s  %-38s A = %-10s kappa = %-10s actual rel = %s' % ('PASS' if good else 'FAIL', name, fnum(R.amp), fnum(R.kappa),
                                                                            fnum(R.actual / abs(R.y) if R.y else math.nan)))
            ok += good
        except Exception as err:
            print('  FAIL  %-38s error %s' % (name, err))
    print('%d of %d passed' % (ok, len(checks)))


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        print()
        sys.exit(0)


def interactive(args):
    print('rounderr   type a formula (plain or LaTeX), or several lines "t1 = ..." ending with the output,')
    print('           finish with an empty line.  Or type a number to pick an example.')
    for k, (name, _, _) in enumerate(EXAMPLES, 1):
        print('  %2d  %s' % (k, name))
    while True:
        print()
        lines = []
        while True:
            s = ask('> ' if not lines else '  ')
            if not s.strip():
                if lines:
                    break
                continue
            lines.append(s)
        text = '\n'.join(lines)
        preset = {}
        if len(lines) == 1 and lines[0].strip().isdigit() and 1 <= int(lines[0]) <= len(EXAMPLES):
            _, text, preset = EXAMPLES[int(lines[0]) - 1]
            print('  ' + text.replace('\n', '\n  '))
        try:
            rd = read_formula(text)
        except Exception as err:
            print('  could not read that (%s). Try again.' % err)
            continue
        print('  I read it as   y = %s' % rd['src'])
        for n, py in rd['script'][:-1]:
            print('                 %s = %s' % (n, py))
        for s in rd['notes']:
            print('  note  %s' % s)
        print('  variables      %s' % (', '.join(rd['vars']) or 'none'))
        if ask('  OK? [Enter = yes, anything else = retype] ').strip():
            continue
        values = {}
        for v in rd['vars']:
            while True:
                s = ask('  %s = %s' % (v, '[%s] ' % preset[v] if v in preset else '')).strip() or preset.get(v, '')
                try:
                    values[v] = s
                    parse_value(s)
                    break
                except Exception as err:
                    print('  could not read that number (%s)' % err)
        s = ask('  precision [double] (double, single, half, or bits) ').strip().lower()
        bits = {'': 53, 'double': 53, 'single': 24, 'half': 11, 'bfloat16': 8}.get(s)
        if bits is None:
            bits = int(s)
        try:
            R = run(text, values, bits, 'as typed')
        except Exception as err:
            print('  analysis failed (%s)' % err)
            continue
        print(text_report(R))
        if args.matrix:
            print(matrix_view(R))
        results = [R]
        while R.candidates:
            s = ask('  pick a candidate number to analyze it, c = compare all, Enter = done ').strip().lower()
            if not s:
                break
            picks = range(1, len(R.candidates) + 1) if s == 'c' else [int(s)]
            for k in picks:
                src, why, _ = R.candidates[k - 1]
                try:
                    R2 = run(src, values, bits, 'candidate %d, %s' % (k, why))
                except Exception as err:
                    print('  candidate failed (%s)' % err)
                    continue
                print(text_report(R2))
                if args.matrix:
                    print(matrix_view(R2))
                results.append(R2)
            if len(results) > 1:
                print(comparison_text(results))
        if args.llm:
            print(llm_opinion('\n\n'.join(text_report(x) for x in results)))
        s = ask('  LaTeX report file name [Enter = skip] ').strip()
        if s:
            if not s.endswith('.tex'):
                s += '.tex'
            print('  ' + write_tex(results, s, pdf=shutil.which('pdflatex') is not None))


def main():
    ap = argparse.ArgumentParser(description='reverse mode rounding error interpreter')
    ap.add_argument('formula', nargs='?', help='formula, or a file name with a tape script')
    ap.add_argument('values', nargs='*', help='name=value pairs, e.g. b=1000 c=1')
    ap.add_argument('--bits', type=int, default=None, help='precision in bits (53 double, 24 single, 11 half)')
    ap.add_argument('--precision', choices=['double', 'single', 'half', 'bfloat16'], default=None)
    ap.add_argument('--compare', action='store_true', help='also analyze every rewrite candidate')
    ap.add_argument('--matrix', action='store_true', help='print the L, B, w, lambda matrix view')
    ap.add_argument('--tex', metavar='FILE', help='write a LaTeX report')
    ap.add_argument('--pdf', action='store_true', help='compile the LaTeX report with pdflatex')
    ap.add_argument('--llm', action='store_true', help='ask an LLM for a plain language second opinion')
    ap.add_argument('--example', type=int, help='run built in example N')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    bits = args.bits or {'double': 53, 'single': 24, 'half': 11, 'bfloat16': 8, None: 53}[args.precision]
    if args.example:
        _, text, vals = EXAMPLES[args.example - 1]
    elif args.formula:
        text = args.formula
        if os.path.isfile(text):
            text = open(text, encoding='utf-8').read()
        vals = {}
        for kv in args.values:
            k, v = kv.split('=', 1)
            vals[k.strip()] = v.strip()
    else:
        interactive(args)
        return
    R = run(text, vals, bits, 'as typed')
    print('I read it as   y = %s' % R.src)
    for s in R.reading['notes']:
        print('note  %s' % s)
    print(text_report(R))
    results = [R]
    if args.matrix:
        print(matrix_view(R))
    if args.compare:
        for k, (src, why, _) in enumerate(R.candidates, 1):
            R2 = run(src, vals, bits, 'candidate %d, %s' % (k, why))
            print(text_report(R2))
            if args.matrix:
                matrix_view(R2)
            results.append(R2)
        if len(results) > 1:
            print(comparison_text(results))
    if args.llm:
        print(llm_opinion('\n\n'.join(text_report(x) for x in results)))
    if args.tex:
        print(write_tex(results, args.tex, pdf=args.pdf))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as err:
        print('rounderr could not finish. %s' % err)
        sys.exit(1)
