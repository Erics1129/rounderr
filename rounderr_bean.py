#!/usr/bin/env python3
"""
rounderr_bean.py   bridge between the rounderr tape engine and Bean
====================================================================

Bean (Kellison, Zielinski, Bindel, Hsu, PLDI 2025) is a small typed language.
If a program type checks, Bean prints a backward error bound for every input
that holds for ALL input values.  https://github.com/Athena-Types/Bean

This file does three things

  1 TRANSLATE   a rounderr tape  ->  a Bean program (.be)
                (or a clear reason why Bean cannot express it)
  2 RUN BEAN    call the bean binary, read the bound of every input (in units of eps)
  3 COMPARE     run the reverse mode engine at your numbers and put both side by side

      Bean bound       static, worst case, every input value        max_j |ln(x_j / x~_j)| <= grade_j * eps
      adjoint ceiling  one backward sweep at these numbers          u * sum_i |lambda_i d_i t_i| / sum_j |g_j x_j|
      measured         what really happened at these numbers        |y_hat - y| / sum_j |g_j x_j|

  (j runs over the inputs Bean is allowed to perturb, the "linear" ones.
   sum_j |g_j x_j| converts an output error into the smallest relative change of
   those inputs that explains it, to first order.)

Translation rules (Bean has only add, sub, mul, div, dmul, and every value that
may carry backward error can be used once)

   a + b, a - b      add / sub, both operands used once
   a * b             mul when both are used once, dmul d x when one of them is discrete
   a / b             div, which returns "number or error", so the rest goes inside a case
   -a                folded into the next add or sub (Bean has no negation)
   x used many times x becomes discrete (dnum for an input, "dlet d = !t" for an
                     intermediate).  Discrete values can only multiply.
   constants         Bean has no literals, so a constant becomes one more input
   sqrt exp sin ...  not expressible, the program is rejected before Bean runs

Usage
   python rounderr_bean.py "((a*x + b)*x + c)*x + d" a=1 b=-3 c=3 d=-1 x=1.0001
   python rounderr_bean.py "x1*y1 + x2*y2" x1=1.5 y1=2.5 x2=-3 y2=1.25 --discrete x1,x2
   python rounderr_bean.py --experiment --tex bean_results.tex --pdf --csv bean_results.csv
   python rounderr_bean.py --bean /path/to/bean.exe ...        (or set the BEAN environment variable)
"""

import argparse
import csv
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import rounderr as rr

def _find_bean():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get('BEAN'), shutil.which('bean'), os.path.join(here, 'bean.exe'), os.path.join(here, 'bean'),
                 os.path.join(here, 'Bean', '_build', 'default', 'bin', 'bean.exe'),
                 '/home/claude/bean_src/_build/default/bin/bean.exe'):
        if cand and os.path.isfile(cand):
            return cand
    return os.environ.get('BEAN') or 'bean'


DEFAULT_BEAN = _find_bean()


class Reject(Exception):
    """the tape cannot be written in Bean"""


# ----------------------------------------------------------------------------
# 1.  TRANSLATE   tape -> Bean program
# ----------------------------------------------------------------------------
def to_bean(tape, force_discrete=()):
    """returns dict(source, linear={bean name: tape line}, discrete={...}, notes=[...], negated=bool)"""
    L = tape.lines
    notes = []
    val = {}                               # tape line -> (root line, sign)
    ops = []                               # dict(idx, op, a, b) on root lines
    for ln in L:
        if ln.op in ('input', 'const'):
            val[ln.idx] = (ln.idx, 1)
            continue
        if ln.op == 'neg':
            r, s = val[ln.args[0]]
            val[ln.idx] = (r, -s)
            continue
        if ln.op == 'sq':
            op, args = 'mul', [ln.args[0], ln.args[0]]
        elif ln.op in ('add', 'sub', 'mul', 'div'):
            op, args = ln.op, ln.args
        else:
            raise Reject('Bean has no primitive for %s' % ln.op)
        (a, sa), (b, sb) = val[args[0]], val[args[1]]
        if op in ('mul', 'div'):
            out_sign = sa * sb
        else:
            if op == 'sub':
                sb = -sb
            if sa == 1 and sb == 1:
                op, out_sign = 'add', 1
            elif sa == 1 and sb == -1:
                op, out_sign = 'sub', 1
            elif sa == -1 and sb == 1:
                op, out_sign, a, b = 'sub', 1, b, a
            else:
                op, out_sign = 'add', -1
        ops.append(dict(idx=ln.idx, op=op, a=a, b=b))
        val[ln.idx] = (ln.idx, out_sign)
    out_root, out_sign = val[tape.out]
    if not ops or ops[-1]['idx'] != out_root:
        raise Reject('the output is not the result of an arithmetic step')
    if out_sign == -1:
        notes.append('the Bean program computes -y (negation is exact, the backward error is the same)')

    # how often is every value used
    uses = {}
    for o in ops:
        uses[o['a']] = uses.get(o['a'], 0) + 1
        uses[o['b']] = uses.get(o['b'], 0) + 1

    def nice(i):
        return tape.label(i)

    forced = {tape.inputs[n] for n in force_discrete if n in tape.inputs}
    discrete = {i for i, k in uses.items() if k > 1} | forced
    for o in ops:                                         # a constant that only scales is discrete too
        for me, other in ((o['a'], o['b']), (o['b'], o['a'])):
            if o['op'] == 'mul' and L[me].op == 'const' and other not in discrete and me != other:
                discrete.add(me)
    for o in ops:
        a, b = o['a'], o['b']
        if o['op'] == 'mul':
            if a == b:
                raise Reject('%s is multiplied by itself, and Bean lets a value that can carry error be used once'
                             % nice(a))
            if a in discrete and b in discrete:
                raise Reject('%s * %s multiplies two reused values, dmul needs one operand that is used once'
                             % (nice(a), nice(b)))
        else:
            for v in (a, b):
                if v in discrete:
                    raise Reject('%s is used more than once and appears in %s, Bean only lets a reused value multiply'
                                 % (nice(v), {'add': 'an addition', 'sub': 'a subtraction', 'div': 'a division'}[o['op']]))
            if a == b:
                raise Reject('%s appears on both sides of one %s' % (nice(a), o['op']))

    # names
    taken = set(tape.inputs)
    names, kcount = {}, 0
    for ln in L:
        if ln.op == 'input':
            names[ln.idx] = ln.name
        elif ln.op == 'const' and ln.idx in uses:
            kcount += 1
            n = 'k%d' % kcount
            while n in taken:
                n += '_'
            names[ln.idx] = n
            taken.add(n)
            notes.append('constant %s became the input %s (Bean has no number literals)' % (ln.name, n))
    for o in ops:
        n = 't%d' % o['idx']
        while n in taken:
            n += '_'
        names[o['idx']] = n
        taken.add(n)

    def ref(i):                                           # how an operand is written
        if i in discrete and L[i].op not in ('input', 'const'):
            return 'd_' + names[i]
        return names[i]

    lin_inputs = {names[i]: i for i in sorted(uses) if L[i].op in ('input', 'const') and i not in discrete}
    dis_inputs = {names[i]: i for i in sorted(uses) if L[i].op in ('input', 'const') and i in discrete}
    for n, i in lin_inputs.items():
        if L[i].op == 'const':
            notes.append('constant %s is added or divided, so Bean may put backward error on it' % n)

    has_div = any(o['op'] == 'div' for o in ops)
    out = ['{' + ' '.join('(%s : dnum)' % n for n in dis_inputs) + '}',
           '{' + ' '.join('(%s : num)' % n for n in lin_inputs) + '}', '']
    closers, ind = [], ''
    for k, o in enumerate(ops):
        last = k == len(ops) - 1
        me = names[o['idx']]
        a, b = o['a'], o['b']
        if o['op'] == 'mul' and (a in discrete or b in discrete):
            d, x = (a, b) if a in discrete else (b, a)
            expr = 'dmul %s %s' % (ref(d), ref(x))
        else:
            expr = '%s %s %s' % (o['op'], ref(a), ref(b))
        if o['op'] == 'div':
            out.append('%slet q_%s = %s;' % (ind, me, expr))
            out.append('%scase q_%s of {' % (ind, me))
            out.append('%s  inl (%s) =>' % (ind, me))
            closers.append('%s| inr (e_%s) => inr num e_%s\n%s}' % (ind, me, me, ind))
            ind += '    '
            if o['idx'] in discrete:
                out.append('%sdlet d_%s = !%s;' % (ind, me, me))
            if last:
                out.append('%sinl () %s' % (ind, me))
        elif last:
            if has_div:
                out.append('%slet %s = %s;' % (ind, me, expr))
                out.append('%sinl () %s' % (ind, me))
            else:
                out.append(ind + expr)
        else:
            out.append('%slet %s = %s;' % (ind, me, expr))
            if o['idx'] in discrete:
                out.append('%sdlet d_%s = !%s;' % (ind, me, me))
    out += reversed(closers)
    return dict(source='\n'.join(out) + '\n', linear=lin_inputs, discrete=dis_inputs,
                notes=notes, negated=out_sign == -1, n_ops=len(ops))


# ----------------------------------------------------------------------------
# 2.  RUN BEAN
# ----------------------------------------------------------------------------
GRADE_RE = re.compile(r"^\s*([A-Za-z_][\w']*) :\[([^\]]+)\]ε", re.M)


_BEAN_CACHE = {}          # sha256 of the Bean program (structure only, no numbers) -> Bean's answer


def run_bean(source, bean=DEFAULT_BEAN, keep=None):
    """returns dict(ok, grades={name: multiple of eps}, message, seconds). Cached by structure hash."""
    key = hashlib.sha256(source.encode()).hexdigest()
    if key in _BEAN_CACHE and keep is None:
        return dict(_BEAN_CACHE[key], cached=True)
    res = _run_bean(source, bean, keep)
    if not res.get('missing'):
        _BEAN_CACHE[key] = res
    return res


def _run_bean(source, bean, keep):
    if not os.path.isfile(bean):
        return dict(ok=False, grades={}, message='bean binary not found at %s' % bean, seconds=0.0, missing=True)
    path = keep
    if path is None:
        fd, path = tempfile.mkstemp(suffix='.be')
        os.close(fd)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(source)
    t0 = time.perf_counter()
    r = subprocess.run([bean, path], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    text = '\n'.join(s for s in (r.stdout + r.stderr).splitlines() if 'stty' not in s)
    if keep is None:
        os.unlink(path)
    if '[TyCheck]' in text or '[Parser' in text or 'Inferred linear context' not in text:
        msg = re.sub(r'\([^()]*\.be:[\d.]+\):\s*(\[\s*\d+\]\s*)?', '', text.strip().splitlines()[0] if text.strip() else 'no output')
        return dict(ok=False, grades={}, message=msg, seconds=dt)
    grades = {m.group(1): float(m.group(2)) for m in GRADE_RE.finditer(text)}
    return dict(ok=True, grades=grades, message=text, seconds=dt)


# ----------------------------------------------------------------------------
# 3.  COMPARE   Bean (static, all inputs)  against  the adjoint engine (these inputs)
# ----------------------------------------------------------------------------
def reference_stored(tape):
    """60 digit value of the formula with the STORED inputs and STORED constants.
    Bean treats constants as inputs, so their storage rounding is not part of the algorithm."""
    vals = []
    for ln in tape.lines:
        if ln.op in ('input', 'const'):
            vals.append(rr.hp(ln.value))
        else:
            a = [vals[j] for j in ln.args]
            if ln.op in ('add', 'sub', 'mul', 'div', 'neg', 'sq', 'abs'):
                r = {'add': lambda: a[0] + a[1], 'sub': lambda: a[0] - a[1], 'mul': lambda: a[0] * a[1],
                     'div': lambda: a[0] / a[1], 'neg': lambda: -a[0], 'sq': lambda: a[0] * a[0],
                     'abs': lambda: abs(a[0])}[ln.op]()
            else:
                r = rr.hp_func(ln.op, *a)
            vals.append(r)
    return vals[tape.out]


def compare(text, values, bits=53, discrete=(), bean=DEFAULT_BEAN, label=''):
    """one row of the experiment. values: name -> string"""
    rd = rr.read_formula(text, list(values))
    rd['typed'] = text
    vals = {v: rr.parse_value(str(values[v])) for v in rd['vars']}
    R = rr.analyze(rd, vals, bits, label)
    return bean_row(R, discrete, bean, label)


def bean_row(R, discrete=(), bean=DEFAULT_BEAN, label=''):
    """Bean against an engine result that already exists"""
    tape, u, bits = R.tape, R.u, R.bits
    row = dict(label=label or R.src, src=R.src, R=R, bits=bits, n_lines=sum(1 for l in tape.lines if l.op not in ('input', 'const')),
               verdict=R.verdict.split('  ')[0], kappa=R.kappa, A=R.amp,
               bean_ok=False, bean_msg='', bean_src='', grades={}, bean_max=None, notes=[],
               eta_meas=None, eta_ceil=None, used=None, ceil_used=None, trials=1, bean_seconds=0.0, cached=False)
    try:
        prog = to_bean(tape, discrete)
    except Reject as err:
        row['bean_msg'] = 'not expressible. ' + str(err)
        row['stage'] = 'translate'
        return row
    row['bean_src'], row['notes'] = prog['source'], prog['notes']
    row['linear'], row['discrete'] = list(prog['linear']), list(prog['discrete'])
    res = run_bean(prog['source'], bean)
    row['raw'] = res['message']
    row['bean_seconds'] = res['seconds']
    row['cached'] = bool(res.get('cached'))
    if not res['ok']:
        row['bean_msg'] = ('Bean rejects it. ' if not res.get('missing') else '') + res['message']
        row['stage'] = 'missing' if res.get('missing') else 'bean'
        return row
    row['bean_ok'] = True
    row['stage'] = 'typed'
    grades = {n: res['grades'].get(n, 0.0) for n in prog['linear']}
    row['grades'] = grades
    row['bean_max'] = max(grades.values()) if grades else 0.0
    # dynamic side, same inputs Bean may perturb
    comp = [l for l in tape.lines if l.op not in ('input', 'const')]
    err = abs(rr.hp_float(rr.hp(R.y) - reference_stored(tape)))
    S = sum(abs(tape.lines[i].lam * tape.lines[i].value) for i in prog['linear'].values())
    W = sum(abs(tape.lines[i].lam * tape.lines[i].value) * grades[n] for n, i in prog['linear'].items())
    ceil = sum(abs(l.lam * l.d * l.value) for l in comp)
    if S > 0:
        row['eta_meas'] = err / S / u                 # measured componentwise backward error, units of u
        row['eta_ceil'] = ceil / S                    # adjoint ceiling for it, units of u
    if W > 0:
        row['used'] = err / u / W                     # share of Bean's allowance actually used (must be <= 1)
        row['ceil_used'] = ceil / W                   # same share for the adjoint worst case
    return row


def worst_of_trials(text, names, discrete, n, rng, bits, bean, label, fixed=None):
    """n random inputs in [-2, 2], keep the trial that uses the largest share of Bean's bound"""
    best, max_ceil, unstable = None, 0.0, 0
    for _ in range(n):
        vals = {v: repr(rng.uniform(-2, 2)) for v in names}
        vals.update(fixed or {})
        r = compare(text, vals, bits, discrete, bean, label)
        if not r['bean_ok']:
            return r
        max_ceil = max(max_ceil, r.get('ceil_used') or 0.0)
        unstable += r['verdict'] != 'STABLE'
        if best is None or (r['used'] or 0) > (best['used'] or 0):
            best = r
    best['trials'], best['max_ceil_used'], best['unstable_trials'] = n, max_ceil, unstable
    return best


# ----------------------------------------------------------------------------
# 4.  the experiment
# ----------------------------------------------------------------------------
def chain_sum(n):
    lines = ['t2 = x1 + x2'] + ['t%d = t%d + x%d' % (k, k - 1, k) for k in range(3, n + 1)]
    lines[-1] = 'y' + lines[-1][lines[-1].index(' ='):]
    return '\n'.join(lines)


def chain_dot(n):
    lines = ['p%d = x%d * y%d' % (k, k, k) for k in range(1, n + 1)]
    lines += ['s2 = p1 + p2'] + ['s%d = s%d + p%d' % (k, k - 1, k) for k in range(3, n + 1)]
    lines[-1] = 'y' + lines[-1][lines[-1].index(' ='):]
    return '\n'.join(lines)


def chain_horner(n):
    """a0 + a1 z + ... + an z^n, n multiplies and n adds"""
    lines = ['m%d = z * a%d' % (n, n), 'h%d = a%d + m%d' % (n - 1, n - 1, n)]
    for k in range(n - 1, 0, -1):
        lines += ['m%d = z * h%d' % (k, k), 'h%d = a%d + m%d' % (k - 1, k - 1, k)]
    lines[-1] = 'y' + lines[-1][lines[-1].index(' ='):]
    return '\n'.join(lines)


def experiment(bean=DEFAULT_BEAN, bits=53):
    rng = random.Random(2026)
    E = []
    add = lambda label, text, vals, disc=(): E.append(compare(text, vals, bits, disc, bean, label))
    add('sum of 4', 'a + b + c + d', {'a': '0.1', 'b': '0.2', 'c': '0.3', 'd': '0.4'})
    add('dot product, 2 terms', 'x1*y1 + x2*y2', {'x1': '1.1', 'y1': '2.3', 'x2': '-0.7', 'y2': '3.1'})
    add('dot product, x fixed', 'x1*y1 + x2*y2', {'x1': '1.1', 'y1': '2.3', 'x2': '-0.7', 'y2': '3.1'}, ('x1', 'x2'))
    add('2 by 2 determinant, nearly singular', 'a*d - b*c', {'a': '1.0000001', 'd': '1.0000002', 'b': '1.0000003', 'c': '1'})
    add('Horner cubic near its triple root', '((a*x + b)*x + c)*x + d', {'a': '1', 'b': '-3', 'c': '3', 'd': '-1', 'x': '1.0001'})
    add('naive cubic', 'a*x*x*x + b*x*x + c*x + d', {'a': '1', 'b': '-3', 'c': '3', 'd': '-1', 'x': '1.0001'})
    add('quotient then add', 'a/b + c', {'a': '1.1', 'b': '3.3', 'c': '-0.3333'})
    add('scaled sum with a constant', '0.1*a + 0.2*b', {'a': '3.7', 'b': '1.9'})
    T = 200
    E.append(worst_of_trials(chain_sum(50), ['x%d' % k for k in range(1, 51)], (), T, rng, bits, bean,
                             'sum of 50, worst of %d random inputs' % T))
    E.append(worst_of_trials(chain_dot(20), ['x%d' % k for k in range(1, 21)] + ['y%d' % k for k in range(1, 21)],
                             tuple('x%d' % k for k in range(1, 21)), T, rng, bits, bean,
                             'dot product of 20, x fixed, worst of %d' % T))
    E.append(worst_of_trials(chain_horner(20), ['a%d' % k for k in range(0, 21)], (), T, rng, bits, bean,
                             'Horner degree 20, worst of %d' % T, fixed={'z': '1.01'}))
    add('x squared minus y squared', 'x**2 - y**2', {'x': '1.0000001', 'y': '1'})
    add('(x - y)(x + y)', '(x - y)*(x + y)', {'x': '1.0000001', 'y': '1'})
    add('quadratic root, textbook', '-b + sqrt(b**2 - c)', {'b': '1000', 'c': '1'})
    add('quadratic root, conjugate', '-c/(b + sqrt(b**2 - c))', {'b': '1000', 'c': '1'})
    add('(1 - cos x)/x squared', '(1 - cos(x))/x**2', {'x': '1e-4'})
    add('a + b - a  (absorption)', 'a + b - a', {'a': '1e15', 'b': '1.3'})
    add('(a + b) c - a c  (unstable, arithmetic only)', '(a + b)*c - a*c', {'a': '1e8', 'b': '1.1', 'c': '3.3'})
    return E


def f(x, digits=3):
    if x is None:
        return ''
    if x == 0:
        return '0'
    if 0.01 <= abs(x) < 1e5:
        return ('%.*g' % (digits, x))
    return rr.fnum(x)


def text_table(E):
    out = ['%-44s %-9s %-10s %-10s %-10s %-9s %s' % ('formula', 'Bean max', 'ceiling', 'measured', 'used', 'engine', 'Bean result'),
           '%-44s %-9s %-10s %-10s %-10s %-9s' % ('', '(eps)', '(u)', '(u)', 'of Bean', 'verdict')]
    for r in E:
        if r['bean_ok']:
            out.append('%-44s %-9s %-10s %-10s %-10s %-9s %s' % (r['label'][:44], f(r['bean_max']), f(r['eta_ceil']), f(r['eta_meas']),
                                                             ('%.0f%%' % (100 * r['used'])) if r['used'] is not None else '', r['verdict'], 'typed'))
        else:
            out.append('%-44s %-9s %-10s %-10s %-10s %-9s %s' % (r['label'][:44], '', '', '', '', r['verdict'], r['bean_msg'][:90]))
    return '\n'.join(out)


def write_csv(E, path):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['label', 'formula', 'tape_lines', 'bean_typed', 'bean_max_grade_eps', 'adjoint_ceiling_u', 'measured_backward_u',
                    'share_of_bean_bound_used', 'engine_verdict', 'kappa', 'A', 'bean_message', 'bean_seconds'])
        for r in E:
            w.writerow([r['label'], r['src'] if len(r['src']) < 80 else '(tape of %d lines)' % r['n_lines'], r['n_lines'], int(r['bean_ok']),
                        r['bean_max'] if r['bean_ok'] else '', r['eta_ceil'] if r['eta_ceil'] is not None else '',
                        r['eta_meas'] if r['eta_meas'] is not None else '', r['used'] if r['used'] is not None else '',
                        r['verdict'], r['kappa'], r['A'], '' if r['bean_ok'] else r['bean_msg'], '%.4f' % r['bean_seconds']])


# ----------------------------------------------------------------------------
# 5.  LaTeX
# ----------------------------------------------------------------------------
def latex(E, single=False):
    T, N = rr.tex_text, rr.tex_num
    P = rr.TexPrinter()
    o = [rr.TEX_HEAD.replace(r'\begin{document}', r'\usepackage{tabularx}' + '\n' + r'\usepackage{listings}' + '\n'
                             + r'\lstset{basicstyle=\ttfamily\small,columns=fullflexible,keepspaces=true,frame=single,framerule=0.3pt,breaklines=true}'
                             + '\n' + r'\begin{document}')]
    w = o.append
    w(r'\section*{Bean and the reverse mode engine, side by side}')
    w(r'\begin{rbox}[title={What is compared}]{askblue}')
    w(r'\textbf{Bean} is static. If the program type checks, every input $x_j$ it may perturb gets a grade $r_j$, and for \emph{all} input values there is '
      r'$\tilde x$ with $|\ln(x_j/\tilde x_j)| \le r_j\,\varepsilon$, $\varepsilon = u/(1-u)$, whose exact result is the computed one.\par'
      r'\textbf{The engine} is dynamic. One backward sweep at the given numbers gives $\lambda$. An output error $\Delta y$ is explained by a relative input change of size '
      r'$|\Delta y| / \sum_j |g_j x_j|$, where $j$ runs over the same inputs Bean may perturb and $g_j = \lambda_j$ on input lines.'
      r'\[ \text{ceiling} = \frac{\sum_i |\lambda_i d_i t_i|}{\sum_j |g_j x_j|}\ u, \qquad \text{measured} = \frac{|\hat y - y|}{\sum_j |g_j x_j|}, \qquad '
      r'\text{used} = \frac{|\hat y - y|}{\varepsilon \sum_j |g_j x_j|\, r_j} \ \ (\text{must stay below } 100\%). \]')
    w(r'\end{rbox}')
    typed = [r for r in E if r['bean_ok']]
    rej = [r for r in E if not r['bean_ok']]
    if typed:
        w(r'\subsection*{Programs Bean accepts}')
        w(r'{\small\begin{tabularx}{\textwidth}{@{}Xrrrrrl@{}}')
        w(r'\toprule formula & steps & Bean max grade ($\varepsilon$) & ceiling ($u$) & measured ($u$) & used & engine \\ \midrule')
        for r in typed:
            w(r'%s & %d & %s & %s & %s & %s & %s \\' % (T(r['label']), r['n_lines'], f(r['bean_max']), f(r['eta_ceil']), f(r['eta_meas']),
                                                        ('%.0f\\%%' % (100 * r['used'])) if r['used'] is not None else '', T(r['verdict'].lower())))
        w(r'\bottomrule\end{tabularx}}')
    if rej:
        w(r'\subsection*{Programs Bean cannot take}')
        w(r'{\small\begin{tabularx}{\textwidth}{@{}p{0.27\textwidth}lX@{}}')
        w(r'\toprule formula & engine verdict & why Bean stops \\ \midrule')
        for r in rej:
            w(r'%s & %s & %s \\' % (T(r['label']), T(r['verdict'].lower()), T(r['bean_msg'])))
        w(r'\bottomrule\end{tabularx}}')
    n_t, n_r = len(typed), len(rej)
    stable_rej = sum(1 for r in rej if r['verdict'] == 'STABLE')
    ok_used = [r['used'] for r in typed if r['used'] is not None]
    w(r'\begin{rbox}[title={Reading the result}]{chipgreen}')
    w(r'\begin{itemize}')
    runs = sum(r.get('trials', 1) for r in typed)
    bad = sum(r.get('unstable_trials', int(r['verdict'] != 'STABLE')) for r in typed)
    w(r'\item Bean accepted %d of %d programs. Over the %d engine runs on accepted programs, the engine verdict was stable in %d.'
      % (n_t, len(E), runs, runs - bad))
    if ok_used:
        w(r'\item The measured backward error %s Bean\textquotesingle s bound. The largest share used was %.0f\%%, the median %.0f\%%.'
          % ('never left' if max(ok_used) <= 1 else r'\textcolor{trapred}{exceeded}', 100 * max(ok_used), 100 * sorted(ok_used)[len(ok_used) // 2]))
    cu = [max(r.get('ceil_used') or 0.0, r.get('max_ceil_used') or 0.0) for r in typed]
    if cu and max(cu) <= 1.0:
        w(r'\item The adjoint ceiling also stayed inside Bean\textquotesingle s bound in every accepted row (largest share %.0f\%%). '
          r'In these runs it matched or beat Bean, which fits the fact that it uses the actual numbers and gives credit for exact steps.' % (100 * max(cu)))
    elif cu:
        w(r'\item The adjoint ceiling reached %.0f\%% of Bean\textquotesingle s bound in the worst accepted row.' % (100 * max(cu)))
    w(r'\item Bean stopped on %d programs. The engine calls %d of them stable and %d unstable, so a rejection by Bean is not a proof of instability.'
      % (n_r, stable_rej, n_r - stable_rej))
    w(r'\end{itemize}')
    w(r'\end{rbox}')
    w(r'\subsection*{Generated Bean programs}')
    for r in E:
        if r['bean_src'] and r['n_lines'] <= 12:
            w(r'\textbf{%s}' % T(r['label']) + (r' \quad $y = %s$' % P(r['src']) if len(r['R'].reading['script']) == 1 else ''))
            w(r'\begin{lstlisting}')
            w(r['bean_src'].rstrip())
            w(r'\end{lstlisting}')
            if r['bean_ok']:
                w('Bean grades. ' + ', '.join('$%s$ gets $%s\\varepsilon$' % (T(n), f(g)) for n, g in r['grades'].items()) + r'.\par')
            else:
                w(r'\textcolor{trapred}{%s}\par' % T(r['bean_msg']))
            for s in r['notes']:
                w(r'{\small Note. %s.}\par' % T(s))
            w(r'\medskip')
    w(r'\end{document}')
    return '\n'.join(o)


def write_tex(E, path, pdf=False):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(latex(E))
    msg = 'wrote ' + path
    if pdf and shutil.which('pdflatex'):
        d = os.path.dirname(os.path.abspath(path)) or '.'
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', os.path.basename(path)],
                           cwd=d, capture_output=True, text=True)
        msg += '  and ' + path[:-4] + '.pdf' if r.returncode == 0 else '  (pdflatex failed, see the .log)'
    return msg


# ----------------------------------------------------------------------------
# 6.  command line
# ----------------------------------------------------------------------------
def show_one(r):
    R = r['R']
    print('formula   y = %s' % (R.src if len(R.reading['script']) == 1 else '(tape of %d lines)' % r['n_lines']))
    if r['bean_src']:
        print('\nBEAN PROGRAM')
        print(r['bean_src'])
    for s in r['notes']:
        print('note  ' + s)
    if r['bean_ok']:
        print('BEAN SAYS   typed, backward stable for every input value')
        for n, g in r['grades'].items():
            print('   %-8s may move by at most %s eps  (%s relative)' % (n, f(g), rr.fnum(g * R.u / (1 - R.u))))
        print('\nENGINE AT YOUR NUMBERS  (units of u, inputs Bean may perturb)')
        print('   adjoint ceiling   %s' % f(r['eta_ceil']))
        print('   measured          %s' % f(r['eta_meas']))
        if r['used'] is not None:
            print('   share of Bean\'s bound actually used   %.0f%%' % (100 * r['used']))
    else:
        print('BEAN SAYS   ' + r['bean_msg'])
    print('   engine verdict    %s   (kappa = %s, A = %s u)' % (R.verdict, rr.fnum(R.kappa), rr.fnum(R.amp)))


def main():
    ap = argparse.ArgumentParser(description='rounderr tape -> Bean, and Bean bound against the adjoint engine')
    ap.add_argument('formula', nargs='?')
    ap.add_argument('values', nargs='*')
    ap.add_argument('--bean', default=DEFAULT_BEAN, help='path to the bean executable')
    ap.add_argument('--discrete', default='', help='comma separated inputs Bean must not perturb')
    ap.add_argument('--bits', type=int, default=53)
    ap.add_argument('--experiment', action='store_true', help='run the built in comparison set')
    ap.add_argument('--tex')
    ap.add_argument('--pdf', action='store_true')
    ap.add_argument('--csv')
    ap.add_argument('--emit', help='write the Bean program to this file')
    args = ap.parse_args()
    if args.experiment:
        E = experiment(args.bean, args.bits)
        print(text_table(E))
    elif args.formula:
        text = args.formula
        if os.path.isfile(text):
            text = open(text, encoding='utf-8').read()
        vals = dict(kv.split('=', 1) for kv in args.values)
        r = compare(text, vals, args.bits, tuple(s for s in args.discrete.split(',') if s), args.bean, 'as typed')
        show_one(r)
        if args.emit and r['bean_src']:
            open(args.emit, 'w', encoding='utf-8').write(r['bean_src'])
        E = [r]
    else:
        ap.print_help()
        return
    if args.csv:
        write_csv(E, args.csv)
        print('wrote ' + args.csv)
    if args.tex:
        print(write_tex(E, args.tex, args.pdf))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, SyntaxError) as err:
        print('rounderr_bean could not finish. %s' % err)
        sys.exit(1)
