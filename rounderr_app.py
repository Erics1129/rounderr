#!/usr/bin/env python3
"""
rounderr_app.py   the whole machine in one window
==================================================

    python rounderr_app.py            opens http://127.0.0.1:8765 in your browser

The same window also runs with no server at all. index.html loads this file into Pyodide
(Python compiled to WebAssembly) inside a web worker, so the machine can be hosted as a
static page, for example on GitHub Pages. bridge() below is the door the worker uses.

Type a formula (plain, LaTeX, a tape, or plain words when a model is switched on).
Everything runs by itself as you type: the analysis, the random trials, the precision
sweep and the input landscape.  The pipeline is

    input -> READ (rules or LLM) -> TAPE -> HASH -> table lookup -> matrix SWEEP -> BEAN -> DATA

    READ    rounderr reader, or an LLM that writes tapes (rounderr_ai)
    TAPE    one line per rounded step
    HASH    sha256 of the tape structure (operations and wiring, no numbers, no names)
    lookup  rounderr_cache.json, a table keyed by that hash. It keeps the sparsity pattern of L,
            Bean's answer and the hit count. A structure seen before skips Bean and recompiling.
    SWEEP   solve L^T lambda = w by back substitution (reverse mode). For random trials the
            compiled tape runs every input at once, each lambda_i is a whole column of numbers.
    BEAN    static backward error bound for all inputs, when Bean can type the program
    DATA    every run is logged, random trials give distributions, all of it exports as CSV and LaTeX

Needs rounderr.py, rounderr_ai.py and rounderr_bean.py in the same folder, plus mpmath.
numpy makes the random trials fast. Bean and an LLM key are optional.

Environment
    BEAN                 path to the bean executable
    ANTHROPIC_API_KEY    for the Claude reader
    OPENAI_BASE_URL      for a local model, default http://localhost:11434/v1
    ROUNDERR_MODEL       model name override
"""

import argparse
import ast
import csv
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rounderr as rr
import rounderr_ai as ra
import rounderr_bean as rb

try:
    import numpy as np
except ImportError:                      # pragma: no cover
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'rounderr_cache.json')


# ----------------------------------------------------------------------------
# 1.  HASH   structure of a tape -> key
# ----------------------------------------------------------------------------
def structure_key(tape):
    """same key for a*b + c and p*q + r.  Numbers and names are not part of it, constants are."""
    parts = []
    for ln in tape.lines:
        if ln.op == 'input':
            parts.append('in')
        elif ln.op == 'const':
            parts.append('c' + ln.name)
        else:
            parts.append(ln.op + ''.join(' %d' % j for j in ln.args))
    text = ';'.join(parts) + '|%d' % tape.out
    return hashlib.sha256(text.encode()).hexdigest()[:12], text


class Table:
    """the hash table. Small JSON file on disk plus compiled tapes in memory."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.compiled = {}                     # key -> FastTape (memory only)
        self.data = {'structures': {}, 'texts': {}}
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
            if isinstance(d, dict) and 'structures' in d:
                self.data = d
                self.data.setdefault('texts', {})
        except (OSError, ValueError):
            pass

    def save(self, force=False):
        """runs are automatic now (one per keystroke), so write at most every two seconds"""
        now = time.time()
        if not force and now - getattr(self, 'saved_at', 0.0) < 2.0:
            return
        self.saved_at = now
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=1)
        except OSError:
            pass

    def touch(self, key, text, tape, src):
        """returns (entry, seen_before)"""
        with self.lock:
            e = self.data['structures'].get(key)
            seen = e is not None
            if not seen:
                comp = [ln for ln in tape.lines if ln.op != 'input']
                e = {'structure': text, 'steps': len(comp), 'nnz': sum(len(ln.args) for ln in comp) + len(comp),
                     'pattern': [[ln.idx, j] for ln in comp for j in ln.args][:400],
                     'first_seen': time.strftime('%Y-%m-%d %H:%M'), 'example': src[:200], 'hits': 0}
                self.data['structures'][key] = e
            e['hits'] = e.get('hits', 0) + 1
            return e, seen


TABLE = Table(CACHE_PATH)


# ----------------------------------------------------------------------------
# 2.  SWEEP, fast version   one compiled tape, all inputs at once
# ----------------------------------------------------------------------------
class FastTape:
    """Forward values, exactness flags and partials as numpy columns, then the back substitution
    for L^T lambda = w down the cached pattern. Matches rounderr.analyze for A, kappa and beta."""

    LIBM = {'exp', 'log', 'log10', 'log2', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh', 'expm1', 'log1p', 'pow'}

    def __init__(self, tape):
        self.ops = [(ln.op, tuple(ln.args), ln.name) for ln in tape.lines]
        self.out = tape.out
        self.inputs = dict(tape.inputs)
        self.const = {ln.idx: (ln.value, ln.d) for ln in tape.lines if ln.op == 'const'}
        self.steps = sum(1 for op, _, _ in self.ops if op not in ('input', 'const'))

    @staticmethod
    def _pow2(a):
        m, _ = np.frexp(np.abs(a))
        return (m == 0.5) & np.isfinite(a)

    @staticmethod
    def _smallint(a):
        return (a == np.floor(a)) & (np.abs(a) < 2.0 ** 26)

    def run(self, X):
        n = len(next(iter(X.values()))) if X else 1
        N = len(self.ops)
        v, d, parts = [None] * N, [0.0] * N, [()] * N
        P2, SI = self._pow2, self._smallint
        with np.errstate(all='ignore'):
            for i, (op, args, name) in enumerate(self.ops):
                if op == 'input':
                    v[i] = np.asarray(X[name], dtype=float)
                    continue
                if op == 'const':
                    v[i] = np.full(n, self.const[i][0])
                    d[i] = self.const[i][1]
                    continue
                a = v[args[0]]
                b = v[args[1]] if len(args) > 1 else None
                if op in ('add', 'sub'):
                    y = b if op == 'add' else -b
                    t = a + y
                    exact = (a == 0) | (y == 0) | ((a * y < 0) & (np.abs(y) / 2 <= np.abs(a)) & (np.abs(a) <= 2 * np.abs(y))) | (SI(a) & SI(b))
                    d[i] = np.where(exact, 0.0, 1.0)
                    parts[i] = (1.0, 1.0 if op == 'add' else -1.0)
                elif op == 'mul':
                    t = a * b
                    d[i] = np.where(P2(a) | P2(b) | (SI(a) & SI(b)), 0.0, 1.0)
                    parts[i] = (b, a)
                elif op == 'div':
                    t = a / b
                    d[i] = np.where(P2(b), 0.0, 1.0)
                    parts[i] = (1.0 / b, -a / (b * b))
                elif op == 'neg':
                    t = -a
                    parts[i] = (-1.0,)
                elif op == 'abs':
                    t = np.abs(a)
                    parts[i] = (np.sign(a),)
                elif op == 'sq':
                    t = a * a
                    d[i] = np.where(P2(a) | SI(a), 0.0, 1.0)
                    parts[i] = (2.0 * a,)
                elif op == 'sqrt':
                    t = np.sqrt(a)
                    d[i] = 1.0
                    parts[i] = (1.0 / (2.0 * t),)
                else:
                    d[i] = 2.0
                    if op == 'exp':
                        t = np.exp(a); parts[i] = (t,)
                    elif op == 'log':
                        t = np.log(a); parts[i] = (1.0 / a,)
                    elif op == 'log10':
                        t = np.log10(a); parts[i] = (1.0 / (a * math.log(10)),)
                    elif op == 'log2':
                        t = np.log2(a); parts[i] = (1.0 / (a * math.log(2)),)
                    elif op == 'sin':
                        t = np.sin(a); parts[i] = (np.cos(a),)
                    elif op == 'cos':
                        t = np.cos(a); parts[i] = (-np.sin(a),)
                    elif op == 'tan':
                        t = np.tan(a); parts[i] = (1.0 + t * t,)
                    elif op == 'asin':
                        t = np.arcsin(a); parts[i] = (1.0 / np.sqrt(1 - a * a),)
                    elif op == 'acos':
                        t = np.arccos(a); parts[i] = (-1.0 / np.sqrt(1 - a * a),)
                    elif op == 'atan':
                        t = np.arctan(a); parts[i] = (1.0 / (1 + a * a),)
                    elif op == 'sinh':
                        t = np.sinh(a); parts[i] = (np.cosh(a),)
                    elif op == 'cosh':
                        t = np.cosh(a); parts[i] = (np.sinh(a),)
                    elif op == 'tanh':
                        t = np.tanh(a); parts[i] = (1.0 - t * t,)
                    elif op == 'expm1':
                        t = np.expm1(a); parts[i] = (t + 1.0,)
                    elif op == 'log1p':
                        t = np.log1p(a); parts[i] = (1.0 / (1.0 + a),)
                    elif op == 'pow':
                        t = np.power(a, b); parts[i] = (b * np.power(a, b - 1), t * np.log(a))
                    else:
                        raise ValueError('no fast rule for %s' % op)
                v[i] = t
            # back substitution for L^T lambda = w
            lam = [0.0] * N
            lam[self.out] = np.ones(n)
            for i in range(N - 1, -1, -1):
                if not parts[i] or (isinstance(lam[i], float) and lam[i] == 0.0):
                    continue
                for j, p in zip(self.ops[i][1], parts[i]):
                    lam[j] = lam[j] + lam[i] * p
            col = lambda z: np.broadcast_to(np.asarray(z, dtype=float), (n,))
            V1, V2 = np.zeros(n), np.zeros(n)
            for i, (op, _, _) in enumerate(self.ops):
                if op == 'input':
                    continue
                w = np.abs(col(lam[i]) * d[i] * v[i])
                V1 += w
                V2 += w * w
            y = v[self.out]
            ay = np.abs(y)
            gx, gn, xn = np.zeros(n), np.zeros(n), np.zeros(n)
            for name, i in self.inputs.items():
                g = col(lam[i])
                gx += np.abs(g * v[i])
                gn += g * g
                xn += v[i] * v[i]
            kappa = gx / ay
            A = V1 / ay
            beta1 = V1 / np.sqrt(gn * xn)
            ratio = A / np.maximum(1.0, kappa)
            valid = np.isfinite(y) & np.isfinite(A) & np.isfinite(kappa) & (ay > 0)
        return dict(y=y, A=A, kappa=kappa, beta1=beta1, ratio=ratio, valid=valid,
                    unstable=ratio > max(10.0, float(self.steps)))


def sample_inputs(base, n, spread, seed):
    """base: name -> float. returns name -> list or array of n values"""
    rng = random.Random(seed)
    out = {}
    for name, x0 in base.items():
        col = []
        for _ in range(n):
            u = rng.uniform(-1, 1)
            if spread == 'pct10':
                x = x0 * (1 + 0.1 * u) if x0 != 0 else 0.1 * u
            else:
                k = 1 if spread == 'x10' else 3
                x = (x0 if x0 != 0 else rng.choice((-1.0, 1.0))) * 10.0 ** (k * u)
            col.append(x)
        out[name] = np.array(col) if np is not None else col
    return out


def run_trials(R, key, n, spread, seed=1):
    base = {name: float(R.values[name]) for name in R.tape.inputs}
    if not base:
        return dict(error='this formula has no inputs to vary')
    fast = np is not None and R.bits == 53
    if not fast:
        n = min(n, 300)
    X = sample_inputs(base, n, spread, seed)
    t0 = time.perf_counter()
    if fast:
        ft = TABLE.compiled.get(key)
        compiled_now = ft is None
        if compiled_now:
            ft = TABLE.compiled[key] = FastTape(R.tape)
        res = ft.run(X)
        y, A, kappa, beta1, ratio = (res[k] for k in ('y', 'A', 'kappa', 'beta1', 'ratio'))
        valid, unstable = res['valid'], res['unstable']
        rows = [i for i in range(n) if valid[i]]
        get = lambda arr, i: float(arr[i])
    else:
        compiled_now = False
        rec = []
        for i in range(n):
            try:
                Ri = rr.analyze(R.reading, {k: Fraction(X[k][i]) for k in base}, R.bits)
                if Ri.y == 0 or not math.isfinite(Ri.amp):
                    raise ValueError
                rec.append((Ri.y, Ri.amp, Ri.kappa, Ri.beta1, Ri.amp / max(1.0, Ri.kappa),
                            Ri.amp > Ri.allow * max(1.0, Ri.kappa)))
            except Exception:
                rec.append(None)
        rows = [i for i in range(n) if rec[i] is not None]
        y, A, kappa, beta1, ratio, unstable = ([r[k] if r else float('nan') for r in rec] for k in range(6))
        get = lambda arr, i: float(arr[i])
    seconds = time.perf_counter() - t0
    if not rows:
        return dict(error='none of the %d random inputs gave a finite nonzero result' % n)
    rs = sorted(get(ratio, i) for i in rows)
    bs = sorted(get(beta1, i) for i in rows)
    med = lambda s: s[len(s) // 2]
    n_un = sum(1 for i in rows if bool(unstable[i]))
    # histogram of log10(ratio), one bar per decade
    logs = [math.log10(max(get(ratio, i), 1e-3)) for i in rows]
    lo, hi = math.floor(min(logs)), math.floor(max(logs))
    hist = [[k, sum(1 for z in logs if math.floor(z) == k)] for k in range(lo, hi + 1)]
    # quarter decade bins for the histogram and a sample of the cloud for the kappa against A plot
    q = {}
    for z in logs:
        q[math.floor(4 * z)] = q.get(math.floor(4 * z), 0) + 1
    bins = [[k / 4.0, q.get(k, 0)] for k in range(min(q), max(q) + 1)]
    stride = max(1, len(rows) // 600)
    points = [[get(kappa, i), get(A, i), int(bool(unstable[i]))] for i in rows[::stride]]
    order = sorted(rows, key=lambda i: -get(ratio, i))
    worst = [dict(inputs={k: float(X[k][i]) for k in base}, y=get(y, i), A=get(A, i), kappa=get(kappa, i),
                  beta1=get(beta1, i), ratio=get(ratio, i), unstable=bool(unstable[i])) for i in order[:5]]
    # confirm the worst one with the full engine (exact slips, 60 digit reference)
    try:
        Rw = rr.analyze(R.reading, {k: Fraction(worst[0]['inputs'][k]) for k in base}, R.bits)
        worst[0]['actual_rel_u'] = abs(Rw.actual) / abs(Rw.y) / Rw.u if Rw.y else None
        worst[0]['ceiling_rel_u'] = Rw.amp
        worst[0]['verdict'] = Rw.verdict.split('  ')[0]
    except Exception:
        pass
    names = list(base)
    table = [[float(X[k][i]) for k in names] + [get(y, i), get(A, i), get(kappa, i), get(beta1, i), get(ratio, i), int(bool(unstable[i]))]
             for i in rows]
    return dict(n=n, valid=len(rows), dropped=n - len(rows), unstable=n_un, unstable_share=n_un / len(rows),
                med_ratio=med(rs), max_ratio=rs[-1], med_beta=med(bs), max_beta=bs[-1],
                seconds=seconds, per_input_us=1e6 * seconds / n, fast=fast, compiled_now=compiled_now,
                hist=hist, bins=bins, points=points, worst=worst, names=names, table=table, spread=spread,
                allow=max(10.0, float(R.steps)))


# ----------------------------------------------------------------------------
# 3.  READ   rules or LLM
# ----------------------------------------------------------------------------
PAIR_RE = re.compile(r'\b([A-Za-z]\w*)\s*=\s*(-?\d[\d.]*(?:[eE][-+]?\d+)?(?:/\d+)?)')


def split_text(text):
    """'formula with b = 1000, c = 1'  ->  ('formula', {'b': '1000', 'c': '1'})"""
    text = text.strip()
    if '\n' in text:
        return text, {}
    m = re.search(r'\b(?:with|where|when|for|at)\b|,\s*(?=[A-Za-z]\w*\s*=\s*-?[\d.])', text)
    if not m:
        return text, {}
    head, tail = text[:m.start()].strip(), text[m.start():]
    pairs = {k: v for k, v in PAIR_RE.findall(tail)}
    return (head, pairs) if head and pairs else (text, {})


def llm_for(mode, model=None):
    if mode == 'anthropic':
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise ValueError('The Claude reader needs the ANTHROPIC_API_KEY environment variable. Set it and restart, or pick another reader.')
        try:
            return ra.AnthropicLLM(model or None)
        except ImportError:
            raise ValueError('The Claude reader needs the anthropic package. Run pip install anthropic.')
    if mode == 'openai':
        return ra.OpenAICompatLLM(model or None)
    return ra.MockLLM()


def read_stage(text, mode, model=None):
    """returns dict(readings=[(name, why, rd)], dropped=[...], reading=str, values={}, notes=[], llm=obj or None, cached=bool)"""
    if mode == 'rule':
        formula, pairs = split_text(text)
        rd = rr.read_formula(formula, list(pairs))
        rd['typed'] = formula
        return dict(readings=[('as typed', '', rd)], dropped=[], reading=rd['src'], values=pairs,
                    notes=rd['notes'], llm=None, cached=False)
    llm = llm_for(mode, model)
    tkey = hashlib.sha256((mode + '|' + (model or '') + '|' + ' '.join(text.split())).encode()).hexdigest()[:16]
    cached = tkey in TABLE.data['texts']
    if cached:
        proposal = TABLE.data['texts'][tkey]
        readings, problems = ra.validate(proposal)
    else:
        proposal, readings, problems = ra.propose(llm, text, lambda *a: None)
        if readings:
            TABLE.data['texts'][tkey] = proposal
    if not readings:
        raise ValueError('The model gave no usable tape. ' + ' | '.join(problems))
    return dict(readings=readings, dropped=[], reading=str(proposal.get('reading', '')),
                values={k: str(v) for k, v in (proposal.get('inputs') or {}).items()},
                notes=problems, llm=llm, cached=cached, proposal=proposal)


# ----------------------------------------------------------------------------
# 4.  JSON shapes
# ----------------------------------------------------------------------------
def clean(o):
    """NaN and infinity are not JSON"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if np is not None and isinstance(o, np.generic):
        return clean(o.item())
    return o


GREEK = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'eps': 'ε', 'epsilon': 'ε', 'theta': 'θ', 'phi': 'φ',
         'psi': 'ψ', 'mu': 'μ', 'nu': 'ν', 'lam': 'λ', 'omega': 'ω', 'sigma': 'σ', 'rho': 'ρ', 'tau': 'τ', 'pi': 'π'}


def mathml(src, name='y'):
    """Python expression source -> presentation MathML, so the window can typeset the formula the way
    it would be written on paper: a real fraction bar, a real root sign, raised exponents."""
    def ident(s):
        m = re.fullmatch(r'([A-Za-z]+?)_?(\d+)', s)
        if m:
            return '<msub>%s<mn>%s</mn></msub>' % (ident(m.group(1)), m.group(2))
        return '<mi>%s</mi>' % GREEK.get(s, s)

    def paren(x):
        return '<mrow><mo>(</mo>%s<mo>)</mo></mrow>' % x

    def go(n, need=0):
        """need: 0 anything, 1 a term of a sum, 2 a factor, 3 an atom"""
        if isinstance(n, ast.Constant):
            v = n.value
            t = repr(v) if not isinstance(v, float) else rr.fnum(v) if abs(v) < 1e-4 or abs(v) >= 1e7 else repr(v)
            m = re.fullmatch(r'(-?[\d.]+)e([-+]?\d+)', t)
            if m:
                return go_sci(m.group(1), int(m.group(2)), need)
            return '<mn>%s</mn>' % t
        if isinstance(n, ast.Name):
            return ident(n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            x = '<mrow><mo>%s</mo>%s</mrow>' % ('−' if isinstance(n.op, ast.USub) else '+', go(n.operand, 2))
            return paren(x) if need >= 2 else x
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Add, ast.Sub)):
                x = '<mrow>%s<mo>%s</mo>%s</mrow>' % (go(n.left, 0), '+' if isinstance(n.op, ast.Add) else '−', go(n.right, 1))
                return paren(x) if need >= 1 else x
            if isinstance(n.op, ast.Mult):
                right = n.right
                dot = isinstance(right, ast.Constant) or isinstance(n.left, ast.Constant) and isinstance(right, ast.Constant) \
                    or (isinstance(n.left, ast.Name) and isinstance(right, ast.Name) and n.left.id == right.id)
                x = '<mrow>%s<mo>%s</mo>%s</mrow>' % (go(n.left, 2), '⋅' if dot else '&#x2062;', go(right, 2))
                return paren(x) if need >= 3 else x
            if isinstance(n.op, ast.Div):
                return '<mfrac>%s%s</mfrac>' % (go(n.left, 0), go(n.right, 0))
            if isinstance(n.op, ast.Pow):
                return '<msup>%s%s</msup>' % (go(n.left, 3), go(n.right, 0))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.args:
            f, a = n.func.id, n.args
            if f == 'sqrt':
                return '<msqrt>%s</msqrt>' % go(a[0], 0)
            if f in ('abs', 'fabs'):
                return '<mrow><mo>|</mo>%s<mo>|</mo></mrow>' % go(a[0], 0)
            if f == 'exp':
                return '<msup><mi>e</mi>%s</msup>' % go(a[0], 0)
            return '<mrow><mi>%s</mi><mo>&#x2061;</mo>%s</mrow>' % ({'log': 'ln', 'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan'}.get(f, f),
                                                                 paren('<mo>,</mo>'.join(go(x, 0) for x in a)))
        raise ValueError('no math for %s' % type(n).__name__)

    def go_sci(mant, e, need):
        x = '<msup><mn>10</mn><mn>%d</mn></msup>' % e
        if mant not in ('1', '1.0'):
            x = '<mrow><mn>%s</mn><mo>×</mo>%s</mrow>' % (mant, x)
            return paren(x) if need >= 3 else x
        return x

    try:
        body = go(ast.parse(src, mode='eval').body)
    except Exception:
        return None
    return '<math><mrow>%s<mo>=</mo>%s</mrow></math>' % (ident(name or 'y'), body)


def result_json(R, label, why=''):
    tape, ay = R.tape, abs(R.y)
    comp = [ln for ln in tape.lines if ln.op != 'input']
    total = sum(abs(ln.lam * ln.d * ln.value) for ln in comp) or 1.0
    lines = []
    for ln in tape.lines:
        w = abs(ln.lam * ln.d * ln.value) if ln.op != 'input' else 0.0
        lines.append(dict(text=ln.text, op=ln.op, value=ln.value, d=ln.d, lam=ln.lam, weight=w, share=w / total,
                          slip=ln.slip, flags=ln.flags, digits_lost=ln.digits_lost if math.isfinite(ln.digits_lost) else None,
                          idx=ln.idx, args=list(ln.args), label=tape.label(ln.idx), name=ln.name,
                          partials=[float(x) for x in ln.partials], slip_u=(ln.slip / (R.u * abs(ln.value))) if ln.value else 0.0))
    rel = R.actual / ay if ay else None
    return dict(label=label, why=why, src=R.src, script=[[n, py] for n, py in R.reading['script']],
                math=[mathml(py, n) for n, py in R.reading['script']],
                multi=len(R.reading['script']) > 1, bits=R.bits, u=R.u,
                inputs={n: float(R.values[n]) for n in tape.inputs}, y=repr(R.y), y_true=rr.mpstr(R.y_true),
                actual=R.actual, rel=rel, rel_u=(rel / R.u) if rel is not None else None, predicted=R.predicted,
                input_effect=R.input_effect, ceiling=R.ceiling, typical=R.typical, ceiling_all1=R.ceiling_all1,
                kappa=R.kappa, A=R.amp, beta1=R.beta1, beta2=R.beta2, verdict=R.verdict,
                word=R.verdict.split('  ')[0] if R.verdict.split('  ')[0] in ('STABLE', 'UNSTABLE') else 'UNDEFINED',
                warnings=R.warnings, diagnosis=R.diagnosis, candidates=[dict(src=c[0], why=c[1]) for c in R.candidates],
                lines=lines, steps=R.steps, allow=R.allow, grad=R.grad, out=tape.out)


def matrix_json(R):
    rr.matrix_view(R)
    M = R.matrix
    n = len(M['lam'])
    if n > 14:
        return dict(n=n, big=True, nnz=sum(1 for row in M['L'] for x in row if x != 0), lam=M['lam'][:0])
    return dict(n=n, big=False, L=M['L'], B=M['B'], w=M['w'], lam=M['lam'], grad=M['grad'],
                tlabels=M['tlabels'], xlabels=M['xlabels'])


def bean_json(R, discrete, bean_path):
    row = rb.bean_row(R, discrete, bean_path)
    return dict(ok=row['bean_ok'], stage=row['stage'], msg=row['bean_msg'], src=row['bean_src'], notes=row['notes'],
                grades=row['grades'], max=row['bean_max'], eta_meas=row['eta_meas'], eta_ceil=row['eta_ceil'],
                used=row['used'], cached=row['cached'], seconds=row['bean_seconds'],
                linear=row.get('linear', []), discrete=row.get('discrete', []), raw=row.get('raw', ''))


# ----------------------------------------------------------------------------
# 4b.  DEVELOPER VIEW   the design, the code behind every stage, and a trace of the run
# ----------------------------------------------------------------------------
STAGES = [
    dict(id='read', title='Read', what='Turn what was typed into Python expression source, one string per tape line. '
         'The rule reader fixes unicode and LaTeX, then walks Python tokens to add the missing * signs, split names like ac, and wrap sin x. '
         'Every guess is recorded as a note. A model reader instead asks an LLM for tapes as JSON, and the engine validates and verifies them.',
         funcs=[('app', 'split_text'), ('app', 'read_stage'), ('rr', 'normalize'), ('rr', 'expand_latex'), ('rr', 'smart_tokens'),
                ('rr', 'read_formula'), ('ra', 'propose'), ('ra', 'validate'), ('ra', 'verify')]),
    dict(id='tape', title='Tape', what='Walk the Python syntax tree once. Every operation that the machine rounds becomes one Line that points at its operand lines. '
         'Inputs come first, constants are lines too because storing 0.1 already rounds.',
         funcs=[('rr', 'Line'), ('rr', 'Tape.from_reading'), ('rr', 'Tape.build'), ('rr', 'Tape.build_pow')]),
    dict(id='hash', title='Hash and table', what='Write the tape as one canonical string of operations and wiring, with no numbers and no names, and take SHA 256 of it. '
         'The first 12 hex digits are the key into rounderr_cache.json. A key seen before reuses the compiled sweep and the stored Bean answer.',
         funcs=[('app', 'structure_key'), ('app', 'Table.touch'), ('rb', 'run_bean')]),
    dict(id='forward', title='Forward pass', what='Run the tape with real machine numbers. For every line also compute the exact result of that one operation on the rounded operands '
         '(Fraction for + - * /, 60 digit mpmath for functions). Slip rho = computed minus exact. The multiplier d is 0 for a provably exact step, 1 for an IEEE step, 2 for a library function. '
         'The local partial derivatives are stored for the way back.',
         funcs=[('rr', 'forward'), ('rr', 'exact_local'), ('rr', 'exactness'), ('rr', 'partials'), ('rr', 'round_to_bits'), ('rr', 'reference')]),
    dict(id='sweep', title='Backward sweep', what='Reverse mode. Put 1 on the output, walk the tape from the last line to the first, and add lambda times the local partial onto each operand. '
         'This is back substitution for L^T lambda = w. The matrix view builds L and B explicitly and checks it. FastTape does the same walk on numpy columns for many inputs at once.',
         funcs=[('rr', 'backward'), ('rr', 'matrix_view'), ('app', 'FastTape.run')]),
    dict(id='numbers', title='Numbers and verdict', what='Combine lambda, d, t and the slips into the reported numbers, decide stable or unstable, explain the blame, and search for rewrites by pattern matching on the syntax tree.',
         funcs=[('rr', 'analyze'), ('rr', 'diagnose'), ('rr', 'rewrite_node'), ('rr', 'find_rewrites')]),
    dict(id='bean', title='Bean', what='Translate the tape into a Bean program and run the Bean type checker. Values used more than once become discrete, negations fold into add or sub, constants become inputs. '
         'Then convert the output error into the smallest relative change of the inputs Bean may perturb, so both tools speak the same unit.',
         funcs=[('rb', 'to_bean'), ('rb', 'run_bean'), ('rb', 'bean_row'), ('rb', 'reference_stored')]),
    dict(id='data', title='Data', what='Log one row per analyzed tape, run random trials through the compiled sweep, and export CSV and LaTeX. The window talks to these functions over a small local HTTP API.',
         funcs=[('app', 'Machine.analyze'), ('app', 'Machine.autofill'), ('app', 'run_trials'), ('app', 'sample_inputs'),
                ('app', 'Machine.precision'), ('app', 'Machine.landscape'), ('app', 'Machine.report_tex'),
                ('app', 'call'), ('app', 'bridge'), ('app', 'make_handler')]),
    dict(id='model', title='Model layer', what='Only when a model reader is on. The model proposes tapes, explains the finished numbers, and may ask the engine to rerun through one tool. '
         'The engine checks the algebra in 60 digits and checks that every number in the explanation exists in the report.',
         funcs=[('ra', 'pipeline'), ('ra', 'explain'), ('ra', 'grounding_check'), ('ra', 'Session.ask'),
                ('ra', 'PROPOSE_SYSTEM'), ('ra', 'EXPLAIN_SYSTEM'), ('ra', 'AGENT_SYSTEM')]),
]
FILES = [('rounderr.py', 'the engine. Reader, tape, forward pass, backward sweep, numbers, rewrites, LaTeX report'),
         ('rounderr_ai.py', 'the model layer. Prompts, proposal, validation, 60 digit verification, explanation, grounding check, follow up agent'),
         ('rounderr_bean.py', 'the Bean bridge. Tape to Bean translation, running Bean, comparison'),
         ('rounderr_app.py', 'this app. Hash table, fast sweep, trials, sweeps, trace, local server, browser bridge'),
         ('index.html', 'the window, markup only'),
         ('rounderr_ui.js', 'the window, behaviour. Auto run, the two engines, tabs, events'),
         ('rounderr_charts.js', 'every picture, drawn by hand as SVG, HTML or canvas. No chart library'),
         ('rounderr.css', 'the look, light and dark'),
         ('engine-worker.js', 'the browser build. Starts Pyodide in a web worker and loads the four Python files into it')]
ROUTES = [('GET /', 'the window'), ('GET /api/status', 'what is installed'), ('GET /api/examples', 'built in examples'),
          ('GET /api/dev', 'this developer view, stages with their source code'),
          ('POST /api/read', 'text and mode in, reading and variable names out'),
          ('POST /api/analyze', 'text, mode, values, bits in. Full result for every candidate out, including the trace'),
          ('POST /api/trials', 'n and spread in. Random trial summary, histogram bins and a cloud sample for every candidate out'),
          ('POST /api/precision', 'the same run at every precision from 8 to 53 bits'),
          ('POST /api/landscape', 'A and kappa along one input, or a map across two'),
          ('POST /api/export', 'run log, trial rows or the LaTeX report as text'),
          ('POST /api/ask', 'question in. Model answer out, plus a new result when the model reran the engine'),
          ('GET /api/log.csv', 'run log'), ('GET /api/trials.csv', 'all trial rows'),
          ('GET /api/report.tex', 'LaTeX report'), ('GET /api/report.pdf', 'compiled report')]
_DEV = {}


def dev_static():
    if _DEV:
        return _DEV
    mods = {'rr': rr, 'ra': ra, 'rb': rb, 'app': sys.modules[__name__]}
    stages = []
    for st in STAGES:
        funcs = []
        for m, qual in st['funcs']:
            obj = mods[m]
            for part in qual.split('.'):
                obj = getattr(obj, part)
            fname = os.path.basename(mods[m].__file__)
            if isinstance(obj, str):
                funcs.append(dict(file=fname, name=qual, line=None, kind='prompt', source=obj))
                continue
            lines, start = inspect.getsourcelines(obj)
            funcs.append(dict(file=fname, name=qual, line=start, kind='code', source=''.join(lines).rstrip('\n')))
        stages.append(dict(id=st['id'], title=st['title'], what=st['what'], funcs=funcs))
    files = []
    for name, role in FILES:
        try:
            with open(os.path.join(HERE, name), encoding='utf-8') as f:
                n = sum(1 for _ in f)
        except OSError:
            n = 0
        files.append(dict(name=name, role=role, lines=n))
    _DEV.update(stages=stages, files=files, routes=[dict(route=a, what=b) for a, b in ROUTES])
    return _DEV


def trace_json(R, key, stext, entry, seen, bean, typed_values):
    """what every stage received and produced on this run, as plain rows and lines"""
    tape, u, F = R.tape, R.u, rr.fnum
    lab = tape.label
    T = {}
    # READ
    rows, notes = [], []
    declared, assigned = set(R.values), []
    for ln in [x for x in (R.reading.get('typed') or R.src).splitlines() if x.strip()]:
        try:
            name, rhs = None, ln
            if '=' in ln and not re.search(r'[<>!=]=', ln):
                name, rhs = [z.strip() for z in ln.split('=', 1)]
            norm = rr.normalize(rhs, notes)
            py = ast.unparse(ast.parse(rr.smart_tokens(norm, declared | set(assigned), notes), mode='eval'))
            rows.append([ln.strip(), norm, (name + ' = ' if name else '') + py])
            if name:
                assigned.append(name)
        except Exception as err:
            rows.append([ln.strip(), '', 'could not replay (%s)' % err])
    tree = ast.dump(ast.parse(R.src, mode='eval').body, indent=1)
    T['read'] = dict(rows=rows, notes=list(dict.fromkeys(notes)), ast=tree[:3000] + ('\n...' if len(tree) > 3000 else ''),
                     values=['%s = %s   typed as %s, stored as %r' % (n, F(float(v)), typed_values.get(n, '?'), tape.lines[tape.inputs[n]].value)
                             for n, v in R.values.items() if n in tape.inputs])
    # TAPE
    T['tape'] = ['%3d  %-6s %-10s %s' % (ln.idx, ln.op, ' '.join(str(j) for j in ln.args) or '.', ln.text) for ln in tape.lines[:120]]
    # HASH
    T['hash'] = dict(structure=stext, sha=hashlib.sha256(stext.encode()).hexdigest(), key=key, seen=seen,
                     entry={k: (v if k != 'pattern' else v[:24]) for k, v in entry.items()})
    # FORWARD
    fw = []
    for ln in tape.lines[:80]:
        exact = ln.exact
        try:
            ex = rr.mpstr(rr.hp(exact) if isinstance(exact, Fraction) else exact)
        except Exception:
            ex = str(exact)[:30]
        default = {0.0: '', 1.0: 'rounded step, slip at most 1 u', 2.0: 'library function, slip at most 2 u'}.get(ln.d, '')
        fw.append(dict(line=ln.text, operands=', '.join('%s = %r' % (lab(j), tape.lines[j].value) for j in ln.args),
                       computed=repr(ln.value), exact=ex, slip=ln.slip, slip_u=(ln.delta / u) if ln.delta else 0.0,
                       d=ln.d, rule=', '.join(ln.flags) or default,
                       partials=', '.join('d/d%s = %s' % (lab(j), F(p)) for j, p in zip(ln.args, ln.partials))))
    T['forward'] = fw
    # SWEEP, replayed with a log
    lam = {ln.idx: 0.0 for ln in tape.lines}
    lam[tape.out] = 1.0
    steps = ['lambda[%s] = 1      start with 1 at the output' % lab(tape.out)]
    for ln in reversed(tape.lines):
        if ln.op in ('input', 'const'):
            continue
        for j, dj in zip(ln.args, ln.partials):
            lam[j] += lam[ln.idx] * dj
            if len(steps) < 90:
                steps.append('lambda[%s] += lambda[%s] * d(%s)/d(%s) = %s * %s      now %s'
                             % (lab(j), lab(ln.idx), lab(ln.idx), lab(j), F(lam[ln.idx]), F(dj), F(lam[j])))
    gap = max(abs(lam[ln.idx] - ln.lam) for ln in tape.lines)
    steps.append('check   this replay and rounderr.backward differ by at most %s' % F(gap))
    T['sweep'] = steps
    # NUMBERS
    comp = [ln for ln in tape.lines if ln.op != 'input']
    v1 = sum(abs(ln.lam * ln.d * ln.value) for ln in comp)
    ay = abs(R.y)
    gx = sum(abs(R.grad[n] * tape.lines[i].value) for n, i in tape.inputs.items())
    T['numbers'] = [
        'u = 2^-%d = %s' % (R.bits, F(u)),
        'actual = computed - true = %r - %s = %s' % (R.y, rr.mpstr(R.y_true), F(R.actual)),
        'predicted = sum lambda_i * rho_i = %s' % F(R.predicted),
        'sum |lambda_i d_i t_i| = %s      (%s)' % (F(v1), ' + '.join(F(abs(ln.lam * ln.d * ln.value)) for ln in comp if ln.d and ln.lam) or '0'),
        'ceiling = u * that sum = %s' % F(R.ceiling),
        'kappa = sum |g_j x_j| / |y| = %s / %s = %s' % (F(gx), F(ay), F(R.kappa)),
        'A = sum |lambda d t| / |y| = %s / %s = %s' % (F(v1), F(ay), F(R.amp)),
        'limit = max(10, rounded steps) = max(10, %d) = %s' % (R.steps, F(R.allow)),
        'stable when A <= limit * max(1, kappa).   %s <= %s * %s is %s' % (F(R.amp), F(R.allow), F(max(1.0, R.kappa)),
                                                                       'true' if R.amp <= R.allow * max(1.0, R.kappa) else 'false'),
        'beta1 = ||v||_1 / (||g||_2 ||x||_2) = %s,   beta2 = %s' % (F(R.beta1), F(R.beta2)),
        'verdict   %s' % R.verdict,
        'rewrite patterns matched   %s' % ('; '.join('%s  ->  %s' % (c[1], c[0]) for c in R.candidates) or 'none (or not searched, a rewrite is only looked for when the verdict is unstable)'),
    ]
    # BEAN
    T['bean'] = dict(discrete=bean.get('discrete', []), linear=bean.get('linear', []), stage=bean['stage'], cached=bean['cached'],
                     raw=bean.get('raw') or bean['msg'], notes=bean['notes'])
    return T


# ----------------------------------------------------------------------------
# 5.  the machine
# ----------------------------------------------------------------------------
class Machine:
    def __init__(self, bean_path):
        self.bean_path = bean_path
        self.lock = threading.Lock()
        self.log = []                 # the experiment data, one row per analyzed tape
        self.results = []             # rr.Result objects of the last run
        self.payload = None
        self.session = None           # ra.Session for follow up questions
        self.trials = None
        self.ctx = None               # what the last run was made from
        self.run_id = 0               # goes up with every analysis, the follow up sweeps quote it
        self.last_sig = None          # the window reruns on every keystroke, log a run only once

    def status(self):
        return dict(numpy=np is not None, mpmath=rr.HAVE_MP, bean=os.path.isfile(self.bean_path), bean_path=self.bean_path,
                    pdflatex=shutil.which('pdflatex') is not None, anthropic_key=bool(os.environ.get('ANTHROPIC_API_KEY')),
                    structures=len(TABLE.data['structures']), log_rows=len(self.log), cache_path=CACHE_PATH,
                    engine='pyodide' if sys.platform == 'emscripten' else 'native',
                    python='%d.%d.%d' % sys.version_info[:3])

    def read(self, req):
        st = read_stage(req.get('text', ''), req.get('mode', 'rule'), req.get('model'))
        variables = []
        for _, _, rd in st['readings']:
            for v in rd['vars']:
                if v not in variables:
                    variables.append(v)
        return dict(reading=st['reading'], vars=variables, values=st['values'], notes=st['notes'],
                    tapes=[dict(name=n, why=w, lines=rd['typed'].splitlines() if rd.get('typed') else [rd['src']]) for n, w, rd in st['readings']])

    def autofill(self, rd, values_text, missing, bits):
        """Numbers for names that have none yet, so a formula runs the moment it is typed.
        Tries a few plain patterns and keeps the first that gives a finite nonzero result."""
        m = len(missing)
        down = ['3', '2', '1.5', '1.25'] + ['%r' % (1 + 2.0 ** -k) for k in range(3, 40)]
        up = ['1.5', '2', '3', '5', '7'] + [str(2 * k + 5) for k in range(3, 40)]
        for pattern in (down[:m], ['1.5'] * m, up[:m], ['0.5'] * m, ['1'] * m, ['0.75', '0.5', '0.25', '0.125'] * 10):
            guess = dict(zip(missing, pattern))
            try:
                vt = dict(values_text, **guess)
                R = rr.analyze(rd, {v: rr.parse_value(vt[v]) for v in rd['vars']}, bits)
                if R.y != 0 and math.isfinite(R.y) and math.isfinite(R.amp):
                    return guess
            except Exception:
                continue
        return None

    def analyze(self, req):
        T = {}
        text, mode, bits = req.get('text', '').strip(), req.get('mode', 'rule'), int(req.get('bits') or 53)
        if not text:
            raise ValueError('Type a formula first.')
        if not 2 <= bits <= 53:
            raise ValueError('Precision must be between 2 and 53 bits.')
        discrete = tuple(req.get('discrete') or ())
        t0 = time.perf_counter()
        st = read_stage(text, mode, req.get('model'))
        T['read'] = time.perf_counter() - t0
        typed = {k: str(v).strip() for k, v in (req.get('values') or {}).items() if str(v).strip()}
        values_text = dict(st['values'])
        values_text.update(typed)
        variables = []
        for _, _, rd in st['readings']:
            for v in rd['vars']:
                if v not in variables:
                    variables.append(v)
        missing = [v for v in variables if v not in values_text]
        autofilled = []
        if missing and req.get('autofill'):
            guess = self.autofill(st['readings'][0][2], values_text, missing, bits)
            if guess:
                values_text.update(guess)
                autofilled, missing = missing, []
        if missing:
            return dict(need_values=missing, vars=variables, values=values_text, reading=st['reading'], notes=st['notes'],
                        text_values=list(st['values']))
        values = {}
        for v in variables:
            try:
                values[v] = rr.parse_value(values_text[v])
            except Exception:
                raise ValueError('%s = %s is not a number I can read.' % (v, values_text[v]))
        readings, dropped = st['readings'], []
        if mode != 'rule':                                   # never trust the model's algebra
            checks = ra.verify(readings, values)
            kept = []
            for (name, why, rd), (ok, note) in zip(readings, checks):
                (kept if ok else dropped).append((name, why, rd) if ok else dict(name=name, why=why, note=note, lines=rd['typed'].splitlines()))
            readings = kept
            if not readings:
                raise ValueError('Every tape from the model failed the 60 digit check. ' + ' | '.join(d['note'] for d in dropped))
        t0 = time.perf_counter()
        results, labels = [], []
        for name, why, rd in readings:
            R = rr.analyze(rd, {v: values[v] for v in rd['vars']}, bits, name)
            results.append(R)
            labels.append((name, why))
        if mode == 'rule' and req.get('compare', True):
            for k, c in enumerate(results[0].candidates, 1):
                try:
                    R2 = rr.run(c[0], values_text, bits, 'rewrite %d' % k, declared=list(values_text))
                    results.append(R2)
                    labels.append(('rewrite %d' % k, c[1]))
                except Exception:
                    pass
        T['sweep'] = time.perf_counter() - t0
        out, t_hash, t_bean = [], 0.0, 0.0
        sig = (text, mode, bits, tuple(sorted(values_text.items())), discrete)
        log_it = sig != self.last_sig
        self.last_sig = sig
        want_trace = req.get('trace', True)
        for R, (name, why) in zip(results, labels):
            t0 = time.perf_counter()
            key, stext = structure_key(R.tape)
            entry, seen = TABLE.touch(key, stext, R.tape, R.src)
            t_hash += time.perf_counter() - t0
            j = result_json(R, name, why)
            j['hash'] = dict(key=key, seen=seen, hits=entry['hits'], nnz=entry['nnz'], steps=entry['steps'], first_seen=entry['first_seen'])
            j['matrix'] = matrix_json(R)
            t0 = time.perf_counter()
            j['bean'] = bean_json(R, discrete, self.bean_path)
            t_bean += time.perf_counter() - t0
            if want_trace:
                j['trace'] = trace_json(R, key, stext, entry, seen, j['bean'], values_text)
            out.append(j)
            if log_it:
                self.log.append(dict(time=time.strftime('%H:%M:%S'), hash=key, seen=int(seen), label=name,
                                     formula=R.src if not j['multi'] else '(tape of %d steps)' % R.steps,
                                     inputs=' '.join('%s=%s' % (n, rr.fnum(float(values[n]))) for n in R.tape.inputs), bits=bits,
                                     y=R.y, actual_rel_u=j['rel_u'], ceiling_rel_u=R.amp, kappa=R.kappa, beta1=R.beta1,
                                     verdict=j['word'], bean=('typed' if j['bean']['ok'] else j['bean']['stage']),
                                     bean_max_eps=j['bean']['max'], bean_used=j['bean']['used'], steps=R.steps))
        T['hash'], T['bean'] = t_hash, t_bean
        TABLE.save()
        narrative, loose = None, []
        self.session = None
        if mode != 'rule':
            t0 = time.perf_counter()
            self.session = ra.Session(st['llm'], text, readings, values, bits, log=lambda *a: None)
            narrative, loose = ra.explain(st['llm'], results, text, log=lambda *a: None)
            T['explain'] = time.perf_counter() - t0
        model_trace = None
        if mode != 'rule':
            model_trace = dict(provider=st['llm'].name, cached=st['cached'], system_prompt='PROPOSE_SYSTEM', user='USER TEXT  ' + text,
                               proposal=st.get('proposal'), problems=st['notes'],
                               checks=[dict(name=n, kept=ok, note=note) for (n, _, _), (ok, note) in zip(st['readings'], checks)],
                               grounding=('every number found in the report' if not loose else 'numbers not in the report: ' + ', '.join(rr.fnum(x) for x in loose)))
        self.results, self.trials = results, None
        self.run_id += 1
        self.ctx = dict(text=text, mode=mode, bits=bits, values_text=values_text, model=req.get('model'), discrete=discrete)
        self.payload = dict(reading=st['reading'], notes=st['notes'], vars=variables, values=values_text, mode=mode, bits=bits,
                            results=out, dropped=dropped, timings={k: 1000 * v for k, v in T.items()},
                            read_cached=st['cached'], narrative=narrative, loose=loose, log_rows=len(self.log),
                            model_trace=model_trace, log_row=self.log[-len(out):],
                            structures=len(TABLE.data['structures']), run_id=self.run_id, autofilled=autofilled,
                            text_values=list(st['values']),
                            total_ms=sum(1000 * v for v in T.values()))
        return self.payload

    def fresh(self, req):
        """the sweeps belong to one analysis. A newer analysis makes an older request stale."""
        if not self.results:
            raise ValueError('Analyze a formula first.')
        if req.get('run_id') is not None and req.get('run_id') != self.run_id:
            raise ValueError('stale')

    def precision(self, req):
        """the same formula and numbers at every precision from 8 to 53 bits"""
        self.fresh(req)
        t0 = time.perf_counter()
        step = max(1, int(req.get('step') or 1))
        want = sorted(set(range(8, 54, step)) | {11, 24, 53, self.ctx['bits']})
        out = []
        for R in self.results:
            rows = []
            for bits in want:
                if bits < 8:
                    continue
                try:
                    Rb = rr.analyze(R.reading, R.values, bits)
                    yt = abs(rr.hp_float(Rb.y_true))        # against the true value, so a result of 0 counts as 100% wrong
                    if not yt or not math.isfinite(Rb.ceiling):
                        continue
                    rows.append([bits, abs(Rb.actual) / yt, Rb.ceiling / yt,
                                 Rb.kappa * Rb.u if math.isfinite(Rb.kappa) else None,
                                 int(Rb.y == 0 or Rb.amp > Rb.allow * max(1.0, Rb.kappa))])
                except Exception:
                    continue
            out.append(dict(label=R.label, rows=rows))
        return dict(precision=out, bits=self.ctx['bits'], ms=1000 * (time.perf_counter() - t0), run_id=self.run_id)

    def landscape(self, req):
        """A and kappa along one input (a curve) or across two inputs (a map), all other inputs fixed.
        One compiled sweep per candidate does every point at once."""
        self.fresh(req)
        if np is None:
            raise ValueError('The landscape needs numpy.')
        t0 = time.perf_counter()
        names = list(self.results[0].tape.inputs)
        if not names:
            return dict(kind='none', run_id=self.run_id)
        base = {n: float(self.results[0].values[n]) for n in names}
        xn = req.get('x') if req.get('x') in names else names[0]
        yn = req.get('y') if req.get('y') in names and req.get('y') != xn else None
        if req.get('y') is None and len(names) > 1 and req.get('kind') != 'curve':
            yn = [n for n in names if n != xn][0]
        decades = min(6.0, max(0.25, float(req.get('decades') or 2)))

        def axis(x0, m):
            e = np.linspace(-decades, decades, m)
            return (e, x0 * 10.0 ** e) if x0 != 0 else (e, np.linspace(-1.0, 1.0, m))

        def sweep(R, X):
            key, _ = structure_key(R.tape)
            ft = TABLE.compiled.get(key)
            if ft is None:
                ft = TABLE.compiled[key] = FastTape(R.tape)
            m = len(next(iter(X.values())))
            full = {n: np.full(m, float(R.values[n])) for n in R.tape.inputs}
            full.update({n: v for n, v in X.items() if n in full})
            return ft.run(full)

        cands = []
        if yn is None:
            m = 201
            ex, xs = axis(base[xn], m)
            for R in self.results:
                res = sweep(R, {xn: xs})
                ok = res['valid']
                dots = []
                if xn in R.tape.inputs:
                    for i in range(4, m, 8):                 # the full engine, exact slips, at a few points
                        try:
                            vals = dict(R.values)
                            vals[xn] = Fraction(float(xs[i]))
                            Ri = rr.analyze(R.reading, vals, R.bits)
                            if Ri.y and math.isfinite(Ri.amp):
                                dots.append([float(ex[i]), abs(Ri.actual) / abs(Ri.y) / Ri.u])
                        except Exception:
                            pass
                cands.append(dict(label=R.label, allow=max(10.0, float(R.steps)), dots=dots,
                                  A=[float(a) if o else None for a, o in zip(res['A'], ok)],
                                  kappa=[float(k) if o else None for k, o in zip(res['kappa'], ok)]))
            return dict(kind='curve', x=xn, y=None, names=names, base=base, decades=decades, e=[float(v) for v in ex],
                        xs=[float(v) for v in xs], log=base[xn] != 0, cands=cands,
                        ms=1000 * (time.perf_counter() - t0), run_id=self.run_id)
        m = 61
        ex, xs = axis(base[xn], m)
        ey, ys = axis(base[yn], m)
        GX, GY = np.meshgrid(xs, ys)
        for R in self.results:
            res = sweep(R, {xn: GX.ravel(), yn: GY.ravel()})
            with np.errstate(all='ignore'):
                z = np.log10(np.maximum(res['ratio'], 1e-3))
            ok = res['valid'] & np.isfinite(z)
            cands.append(dict(label=R.label, allow=max(10.0, float(R.steps)),
                              unstable_share=float(np.mean(res['unstable'][ok])) if ok.any() else None,
                              z=[round(float(v), 2) if o else None for v, o in zip(z, ok)]))
        return dict(kind='map', x=xn, y=yn, names=names, base=base, decades=decades, m=m,
                    xs=[float(v) for v in xs], ys=[float(v) for v in ys], xlog=base[xn] != 0, ylog=base[yn] != 0,
                    cands=cands, ms=1000 * (time.perf_counter() - t0), run_id=self.run_id)

    def export(self, req):
        """the downloads as text, so the page can save them with or without a server"""
        kind = req.get('kind')
        if kind == 'log':
            return dict(name='rounderr_log.csv', mime='text/csv', text=self.log_csv())
        if kind == 'trials':
            return dict(name='rounderr_trials.csv', mime='text/csv', text=self.trials_csv())
        if kind == 'tex':
            return dict(name='rounderr_report.tex', mime='application/x-tex', text=self.report_tex())
        raise ValueError('nothing to export under that name')

    def run_trials(self, req):
        self.fresh(req)
        n = max(10, min(int(req.get('n') or 1000), 20000))
        spread = req.get('spread') or 'x10'
        rows = []
        for R in self.results:
            key, _ = structure_key(R.tape)
            t = run_trials(R, key, n, spread, seed=int(req.get('seed') or 1))
            t['label'] = R.label
            rows.append(t)
        self.trials = rows
        return dict(trials=[{k: v for k, v in t.items() if k != 'table'} for t in rows], run_id=self.run_id)

    def ask(self, req):
        if self.session is None:
            raise ValueError('Follow up questions need a model reader. Pick one in the reader menu and analyze again.')
        before = (dict(self.session.values), self.session.bits)
        answer = self.session.ask(req.get('question', ''))
        payload = None
        if (dict(self.session.values), self.session.bits) != before:
            vt = dict(self.ctx['values_text'])
            vt.update({k: ('%d' % v if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))) for k, v in self.session.values.items()})
            payload = self.analyze(dict(text=self.ctx['text'], mode=self.ctx['mode'], bits=self.session.bits,
                                        values=vt, model=self.ctx['model'], discrete=self.ctx['discrete']))
        return dict(answer=answer, payload=payload)

    # ---- exports -------------------------------------------------------------
    def log_csv(self):
        buf = io.StringIO()
        cols = ['time', 'hash', 'seen', 'label', 'formula', 'inputs', 'bits', 'steps', 'y', 'actual_rel_u', 'ceiling_rel_u',
                'kappa', 'beta1', 'verdict', 'bean', 'bean_max_eps', 'bean_used']
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in self.log:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in cols})
        return buf.getvalue()

    def trials_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf)
        if not self.trials:
            return 'run random trials first\n'
        for t in self.trials:
            if 'table' not in t:
                continue
            w.writerow(['candidate'] + t['names'] + ['y', 'A', 'kappa', 'beta1', 'A_over_kappa', 'unstable'])
            for row in t['table']:
                w.writerow([t['label']] + row)
        return buf.getvalue()

    def report_tex(self):
        if not self.results:
            raise ValueError('Analyze a formula first.')
        tex = rr.latex_report(self.results, title='Rounding error report')
        T, N = rr.tex_text, rr.tex_num
        extra = []
        beans = [r['bean'] for r in self.payload['results']]
        extra.append(r'\section*{Bean, static backward error for all inputs}')
        extra.append(r'\begin{tabular}{@{}llll@{}} \toprule candidate & Bean & largest grade ($\varepsilon$) & share of the bound used here \\ \midrule')
        for R, b in zip(self.results, beans):
            extra.append(r'%s & %s & %s & %s \\' % (T(R.label), 'typed' if b['ok'] else T(b['msg'][:70]),
                                                 ('$%s$' % N(b['max'])) if b['ok'] else '',
                                                 ('%.0f\\%%' % (100 * b['used'])) if b.get('used') is not None else ''))
        extra.append(r'\bottomrule \end{tabular}')
        if self.trials:
            extra.append(r'\section*{Random trials}')
            extra.append(r'\begin{tabular}{@{}lrrrrr@{}} \toprule candidate & inputs & unstable & median $A/\kappa$ & largest $A/\kappa$ & largest $\beta_1$ \\ \midrule')
            for t in self.trials:
                if 'error' in t:
                    continue
                extra.append(r'%s & %d & %.1f\%% & $%s$ & $%s$ & $%s$ \\' % (T(t['label']), t['valid'], 100 * t['unstable_share'],
                                                                            N(t['med_ratio']), N(t['max_ratio']), N(t['max_beta'])))
            extra.append(r'\bottomrule \end{tabular}')
        if self.payload.get('narrative'):
            extra.append(r'\section*{The model explains}')
            extra += [T(p) + r'\par\medskip' for p in self.payload['narrative'].strip().split('\n\n')]
        return tex.replace(r'\end{document}', '\n'.join(extra) + '\n' + r'\end{document}')

    def report_pdf(self):
        if not shutil.which('pdflatex'):
            raise ValueError('pdflatex is not installed on this computer. Download the .tex file instead.')
        d = tempfile.mkdtemp(prefix='rounderr_')
        with open(os.path.join(d, 'report.tex'), 'w', encoding='utf-8') as f:
            f.write(self.report_tex())
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'report.tex'], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            raise ValueError('pdflatex could not compile the report.')
        with open(os.path.join(d, 'report.pdf'), 'rb') as f:
            data = f.read()
        shutil.rmtree(d, ignore_errors=True)
        return data


# ----------------------------------------------------------------------------
# 6.  one door (dispatch), the browser build (bridge), the window and the local server
# ----------------------------------------------------------------------------
PAGE = 'index.html'
STATIC = {'index.html': 'text/html', 'rounderr.css': 'text/css', 'rounderr_ui.js': 'text/javascript',
          'rounderr_charts.js': 'text/javascript', 'engine-worker.js': 'text/javascript', 'snapshot.json': 'application/json',
          'rounderr.py': 'text/plain', 'rounderr_ai.py': 'text/plain', 'rounderr_bean.py': 'text/plain',
          'rounderr_app.py': 'text/plain', 'rounderr_design.pdf': 'application/pdf'}
POSTS = {'read': 'read', 'analyze': 'analyze', 'trials': 'run_trials', 'precision': 'precision',
         'landscape': 'landscape', 'ask': 'ask', 'export': 'export'}


def call(machine, route, req=None):
    """Every request goes through here, from the local server and from the browser build alike.
    Returns a plain dict. Errors come back as dict(error=...) so the window stays alive."""
    try:
        with machine.lock:
            if route == 'status':
                return machine.status()
            if route == 'examples':
                return [dict(name=n, text=t, values=v) for n, t, v in rr.EXAMPLES]
            if route == 'dev':
                return dev_static()
            if route not in POSTS:
                return dict(error='not found')
            return getattr(machine, POSTS[route])(req or {})
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, KeyError) as err:
        msg = str(err) if not isinstance(err, SyntaxError) else 'I could not read that formula. Check the brackets and operators.'
        return dict(error=msg or err.__class__.__name__)
    except Exception as err:              # keep the window alive, show what broke
        if err.__class__.__name__ == 'TokenError':            # half typed, the page reruns on every keystroke
            return dict(error='The formula is not finished yet. A bracket is still open.')
        return dict(error='%s. %s' % (err.__class__.__name__, err))


_BROWSER = {}


def bridge(route, body='{}'):
    """The browser build. engine-worker.js imports this module inside Pyodide and calls bridge()
    with a route name and a JSON string, exactly what the local server would have received."""
    if 'machine' not in _BROWSER:
        _BROWSER['machine'] = Machine(rb.DEFAULT_BEAN)
    return json.dumps(clean(call(_BROWSER['machine'], route, json.loads(body or '{}'))))


def use_numpy(module):
    """the browser build starts answering before numpy has arrived, then hands it over here"""
    global np
    np = module
    return True


def make_handler(machine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype='application/json', name=None):
            data = body if isinstance(body, bytes) else body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype + ('; charset=utf-8' if ctype.startswith('text') or ctype.endswith('json') else ''))
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            if name:
                self.send_header('Content-Disposition', 'attachment; filename="%s"' % name)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split('?')[0]
            name = PAGE if path == '/' else path.lstrip('/')
            if name in STATIC:
                try:
                    with open(os.path.join(HERE, name), 'rb') as f:
                        return self.send(200, f.read(), STATIC[name])
                except OSError:
                    return self.send(404, 'not found', 'text/plain')
            if path == '/favicon.ico':
                return self.send(204, b'', 'image/x-icon')
            if path in ('/api/status', '/api/dev', '/api/examples'):
                return self.send(200, json.dumps(clean(call(machine, path[5:]))))
            if path in ('/api/log.csv', '/api/trials.csv', '/api/report.tex'):
                r = call(machine, 'export', dict(kind={'log.csv': 'log', 'trials.csv': 'trials', 'report.tex': 'tex'}[path[5:]]))
                return self.send(200, r['text'], r['mime'], r['name']) if 'text' in r else self.send(400, r['error'], 'text/plain')
            if path == '/api/report.pdf':
                try:
                    with machine.lock:
                        return self.send(200, machine.report_pdf(), 'application/pdf', 'rounderr_report.pdf')
                except Exception as err:
                    return self.send(400, str(err), 'text/plain')
            self.send(404, 'not found', 'text/plain')

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            try:
                req = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                return self.send(400, json.dumps(dict(error='bad request')))
            route = self.path[5:] if self.path.startswith('/api/') else ''
            if route not in POSTS:
                return self.send(404, json.dumps(dict(error='not found')))
            self.send(200, json.dumps(clean(call(machine, route, req))))
    return H


DEFAULT = dict(text='-b + sqrt(b^2 - c)', values={'b': '1000', 'c': '1'})


def snapshot(path=None):
    """What the default example returns, saved next to the page. The hosted page paints this at once
    while Python is still starting in the browser, then swaps in live answers."""
    M = Machine(rb.DEFAULT_BEAN)
    snap = dict(DEFAULT)
    snap['analyze'] = call(M, 'analyze', dict(text=DEFAULT['text'], mode='rule', values=DEFAULT['values'], autofill=True))
    rid = snap['analyze']['run_id']
    snap['trials'] = call(M, 'trials', dict(run_id=rid, n=2000, spread='x10'))
    snap['precision'] = call(M, 'precision', dict(run_id=rid))
    snap['landscape'] = call(M, 'landscape', dict(run_id=rid))
    snap['examples'] = call(M, 'examples')
    snap['status'] = dict(call(M, 'status'), engine='snapshot', bean=False, pdflatex=False, anthropic_key=False)
    path = path or os.path.join(HERE, 'snapshot.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean(snap), f, separators=(',', ':'))
    print('wrote %s (%d bytes)' % (path, os.path.getsize(path)))


def selftest():
    print('rounderr_app self test')
    ok = 0
    # 1 the fast sweep equals the full engine
    worst = 0.0
    for name, text, vals in rr.EXAMPLES:
        if not vals:
            continue
        R = rr.run(text, vals)
        res = FastTape(R.tape).run({k: np.array([float(R.values[k])]) for k in R.tape.inputs})
        for a, b in ((res['A'][0], R.amp), (res['kappa'][0], R.kappa), (res['beta1'][0], R.beta1)):
            worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    good = worst < 1e-9
    print('  %s  fast matrix sweep equals the full engine on the examples (largest gap %.1e)' % ('PASS' if good else 'FAIL', worst))
    ok += good
    # 2 same structure, different names and numbers, same hash
    k1 = structure_key(rr.run('a*b + c', {'a': '1', 'b': '2', 'c': '3'}).tape)[0]
    k2 = structure_key(rr.run('p*q + r', {'p': '7', 'q': '8', 'r': '9'}).tape)[0]
    k3 = structure_key(rr.run('a*b - c', {'a': '1', 'b': '2', 'c': '3'}).tape)[0]
    good = k1 == k2 and k1 != k3
    print('  %s  hash ignores names and numbers, changes with the structure (%s, %s, %s)' % ('PASS' if good else 'FAIL', k1, k2, k3))
    ok += good
    # 3 the machine end to end
    M = Machine(rb.DEFAULT_BEAN)
    p = M.analyze(dict(text='-b + sqrt(b**2 - c) with b = 1000, c = 1', mode='rule'))
    good = p['results'][0]['word'] == 'UNSTABLE' and p['results'][1]['word'] == 'STABLE'
    print('  %s  typed text with numbers inside, textbook form unstable, rewrite stable' % ('PASS' if good else 'FAIL'))
    ok += good
    t = M.run_trials(dict(n=2000, spread='x10'))['trials']
    good = t[0]['unstable_share'] > 0.5 and t[1]['unstable_share'] == 0
    print('  %s  2000 random inputs, textbook unstable in %.0f%%, rewrite in %.0f%% (%.2f microseconds per input)'
          % ('PASS' if good else 'FAIL', 100 * t[0]['unstable_share'], 100 * t[1]['unstable_share'], t[0]['per_input_us']))
    ok += good
    p = M.analyze(dict(text='positive root of x^2 + 2bx + c = 0 with b = 1000, c = 1', mode='mock'))
    good = len(p['results']) == 2 and p['narrative'] and not p['loose']
    print('  %s  model reader (offline stand in), two tapes verified, explanation grounded' % ('PASS' if good else 'FAIL'))
    ok += good
    json.dumps(clean(p), allow_nan=False)
    good = len(M.log_csv().splitlines()) == 5 and 'documentclass' in M.report_tex()
    print('  %s  data log has 4 rows, LaTeX report builds' % ('PASS' if good else 'FAIL'))
    ok += good
    rid = p['run_id']
    pr, ls = M.precision(dict(run_id=rid)), M.landscape(dict(run_id=rid))
    stale = call(M, 'trials', dict(run_id=rid - 1)).get('error') == 'stale'
    good = len(pr['precision'][0]['rows']) > 40 and ls['kind'] == 'map' and len(ls['cands'][0]['z']) == ls['m'] ** 2 and stale
    print('  %s  precision sweep, two input map, and an old run id is refused as stale' % ('PASS' if good else 'FAIL'))
    ok += good
    p = M.analyze(dict(text='(1 - cos(x))/x^2', mode='rule', autofill=True))
    good = p.get('autofilled') == ['x'] and M.landscape(dict(run_id=p['run_id']))['kind'] == 'curve'
    print('  %s  a formula with no numbers gets numbers picked for it, one input gives a curve' % ('PASS' if good else 'FAIL'))
    ok += good
    json.dumps(clean(dict(a=pr, b=ls)), allow_nan=False)
    print('%d of 8 passed' % ok)


def print_trace(text, bean_path):
    """the Developer tab as plain text"""
    p = Machine(bean_path).analyze(dict(text=text, mode='rule', compare=False))
    if 'need_values' in p:
        print('give numbers in the text, for example  "%s with %s"' % (text, ', '.join('%s = 1' % v for v in p['need_values'])))
        return
    r = p['results'][0]
    t = r['trace']
    what = {s['id']: s for s in STAGES}
    def head(i):
        print('\n' + '=' * 78 + '\n%s\n%s' % (what[i]['title'].upper(), what[i]['what']))
        print('code   ' + ', '.join('%s.%s' % (m, q) for m, q in what[i]['funcs']) + '\n' + '-' * 78)
    head('read')
    for a, b, c in t['read']['rows']:
        print('typed       %s\nnormalized  %s\npython      %s' % (a, b, c))
    for n in t['read']['notes'] + t['read']['values']:
        print('  ' + n)
    head('tape')
    print('idx  op     args       line')
    print('\n'.join(t['tape']))
    head('hash')
    print('structure  %s\nsha256     %s\nkey        %s   %s' % (t['hash']['structure'], t['hash']['sha'], t['hash']['key'], 'seen before' if t['hash']['seen'] else 'new'))
    head('forward')
    for f in t['forward']:
        print('%-22s computed %-24s exact %-28s slip %-10s d %g  %s' % (f['line'], f['computed'], f['exact'], rr.fnum(f['slip']), f['d'], f['rule']))
    head('sweep')
    print('\n'.join(t['sweep']))
    head('numbers')
    print('\n'.join(t['numbers']))
    head('bean')
    print('discrete %s   linear %s   %s' % (t['bean']['discrete'], t['bean']['linear'], t['bean']['stage']))
    print(r['bean']['src'] or '')
    print(t['bean']['raw'])
    head('data')
    print(json.dumps(clean(p['log_row'][0]), indent=1))


def main():
    ap = argparse.ArgumentParser(description='rounderr, the whole machine in one window')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--bean', default=rb.DEFAULT_BEAN, help='path to the bean executable')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--snapshot', action='store_true', help='rewrite snapshot.json, the instant first paint of the hosted page')
    ap.add_argument('--trace', metavar='TEXT', help='print the developer trace of one run in the terminal and exit')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.snapshot:
        snapshot()
        return
    if args.trace:
        print_trace(args.trace, args.bean)
        return
    machine = Machine(args.bean)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(machine))
            break
        except OSError:
            continue
    if server is None:
        print('no free port near %d' % args.port)
        return
    url = 'http://127.0.0.1:%d' % server.server_address[1]
    print('rounderr is running at %s   (Ctrl C to stop)' % url)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    TABLE.save(force=True)


if __name__ == '__main__':
    main()
