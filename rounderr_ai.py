#!/usr/bin/env python3
"""
rounderr_ai.py   the AI powered layer on top of rounderr.py
=============================================================

Structure (LLM proposes, the tape engine disposes)

   you type anything            "positive root of x^2 + 2bx + c = 0 with b = 1000, c = 1"
        |
   [1] PROPOSE  (LLM)           reads it, writes 1 to 3 tapes in the tape language as JSON
        |                       tape 1 = your formula as written, others = equal algebra, maybe more stable
   [2] VALIDATE (engine)        every line parsed by rounderr's reader, errors go back to the LLM once
        |
   [3] VERIFY   (engine)        all tapes evaluated in 60 digits at your numbers and at perturbed numbers,
        |                       a tape that disagrees is wrong algebra and is dropped (never trust the LLM's math)
   [4] ANALYZE  (engine)        forward pass, slips, backward sweep, lambda, ceiling, kappa, A, beta, verdict
        |
   [5] EXPLAIN  (LLM)           plain words from the numbers, then a grounding check that every number it
        |                       wrote exists in the report
   [6] FOLLOW UP (LLM + engine) "what if b = 1e6", "try single precision", "why is t5 to blame"
                                the LLM answers from the report or asks the engine to run again (one tool)

The LLM never does arithmetic.  The engine never guesses what you meant.

Existing LLM ideas used
   program aided LMs / tool use     the model writes the tape, the interpreter runs it
   structured output (JSON contract) one fixed schema in, one fixed schema out
   few shot prompting               three worked examples of the contract in the prompt
   self refine                      validation and verification errors are fed back once
   verification by execution        algebra is checked numerically, not believed
   grounded generation + check      narrative numbers must appear in the engine's report
   ReAct style agent loop           answer, or call the one tool (run) and answer after

Providers
   --provider anthropic   needs  pip install anthropic  and  ANTHROPIC_API_KEY   (model via --model)
   --provider openai      any OpenAI compatible endpoint (Ollama, vLLM, OpenAI) via --base-url
   --provider mock        offline stand in that follows the same contract, for tests and demos
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from fractions import Fraction

import rounderr as rr

# ----------------------------------------------------------------------------
# 0.  LLM clients  (one method: complete(system, user) -> text)
# ----------------------------------------------------------------------------
class AnthropicLLM:
    name = 'anthropic'

    def __init__(self, model=None):
        import anthropic                                   # pip install anthropic
        self.client = anthropic.Anthropic()                # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get('ROUNDERR_MODEL', 'claude-sonnet-5-5')

    def complete(self, system, user):
        resp = self.client.messages.create(model=self.model, max_tokens=3000, temperature=0,
                                           system=system,
                                           messages=[{'role': 'user', 'content': user}])
        return ''.join(getattr(b, 'text', '') for b in resp.content)


class OpenAICompatLLM:
    name = 'openai'

    def __init__(self, model=None, base_url=None):
        self.model = model or os.environ.get('ROUNDERR_MODEL', 'llama3.1')
        self.base_url = (base_url or os.environ.get('OPENAI_BASE_URL', 'http://localhost:11434/v1')).rstrip('/')
        self.key = os.environ.get('OPENAI_API_KEY', '')

    def complete(self, system, user):
        body = json.dumps({'model': self.model, 'temperature': 0,
                           'messages': [{'role': 'system', 'content': system},
                                        {'role': 'user', 'content': user}]}).encode()
        req = urllib.request.Request(self.base_url + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': 'Bearer ' + self.key})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode())
        return data['choices'][0]['message']['content']


class MockLLM:
    """offline stand in. It obeys the same contract, so the whole pipeline runs without a key."""
    name = 'mock'

    def complete(self, system, user):
        if system.startswith('ROLE PROPOSE'):
            return json.dumps(self.propose(user))
        if system.startswith('ROLE EXPLAIN'):
            return self.explain(user)
        if system.startswith('ROLE AGENT'):
            return json.dumps(self.agent(user))
        return '{}'

    @staticmethod
    def pairs(text):
        vals = {}
        for m in re.finditer(r'\b([A-Za-z]\w*)\s*=\s*(-?\d[\d.]*(?:[eE][-+]?\d+)?|\d+/\d+)', text):
            vals[m.group(1)] = m.group(2)
        return vals

    def propose(self, user):
        text = user.split('USER TEXT')[-1]
        low = text.lower()
        inputs = self.pairs(text)
        if 'quadratic' in low or '2bx' in low or 'root' in low:
            inputs = inputs or {'b': '1000', 'c': '1'}
            return {'reading': 'the root -b + sqrt(b^2 - c) of x^2 + 2bx + c = 0', 'inputs': inputs, 'missing': [],
                    'tapes': [{'name': 'textbook formula',
                               'lines': ['t1 = b*b', 't2 = t1 - c', 't3 = sqrt(t2)', 'y = -b + t3'],
                               'why': 'as written'},
                              {'name': 'conjugate form',
                               'lines': ['t1 = b*b', 't2 = t1 - c', 't3 = sqrt(t2)', 't4 = b + t3', 'y = -c / t4'],
                               'why': 'multiply top and bottom by b + sqrt(b^2 - c), no cancellation'}]}
        if 'cos' in low:
            inputs = inputs or {'x': '1e-4'}
            return {'reading': '(1 - cos x) / x^2', 'inputs': inputs, 'missing': [],
                    'tapes': [{'name': 'as written',
                               'lines': ['t1 = cos(x)', 't2 = 1 - t1', 't3 = x*x', 'y = t2 / t3'], 'why': 'as written'},
                              {'name': 'half angle',
                               'lines': ['t1 = x/2', 't2 = sin(t1)', 't3 = t2*t2', 't4 = 2*t3', 't5 = x*x', 'y = t4 / t5'],
                               'why': '1 - cos x = 2 sin^2(x/2), no cancellation'}]}
        if 'wrong' in low:                       # used by the self test, deliberately bad algebra
            inputs = inputs or {'x': '0.5'}
            return {'reading': 'x*x + 1', 'inputs': inputs, 'missing': [],
                    'tapes': [{'name': 'as written', 'lines': ['t1 = x*x', 'y = t1 + 1'], 'why': 'as written'},
                              {'name': 'bad rewrite', 'lines': ['t1 = x + 1', 'y = t1*t1'], 'why': 'looks equal, is not'}]}
        formula = re.split(r'\bwith\b|\bat\b|,|;', text, maxsplit=1)[0].strip()
        rd = rr.read_formula(formula, list(inputs))
        return {'reading': formula, 'inputs': inputs, 'missing': [v for v in rd['vars'] if v not in inputs],
                'tapes': [{'name': 'as written', 'lines': ['y = ' + rd['src']], 'why': 'as written'}]}

    def explain(self, user):
        g = lambda pat: (re.search(pat, user) or [None, '?'])[1]
        actual = g(r'actual error\s+(\S+)')
        ceiling = g(r'ceiling\s+(\S+)')
        A = g(r'algorithm ceiling\s+A = (\S+)')
        kappa = g(r'kappa = (\S+)')
        verdict = g(r'verdict\s+(\w+)')
        blame = g(r'\* line (\w+) carries')
        return ('What actually happened. The computed value is off by %s, and the first order sum of the measured slips predicts the same number.\n\n'
                'What the error might be. The worst case ceiling is %s, which is %s u relative, while the problem itself only forces about %s u.\n\n'
                'Who is to blame and why. Line %s carries the ceiling. Its slip is kept while the following subtraction throws away the good digits.\n\n'
                'What to do. The verdict is %s. Pick the candidate with the smallest ceiling in the comparison table.'
                % (actual, ceiling, A, kappa, blame, verdict))

    def agent(self, user):
        q = user.split('QUESTION')[-1].lower()
        pairs = self.pairs(q)
        if 'after your run' in user:                   # the tool already ran, now answer from the new report
            prec = (re.search(r'precision (\d+) bits', user) or [None, '?'])[1]
            verdicts = re.findall(r'verdict\s+(UNSTABLE|STABLE|output)', user)
            return {'action': 'answer', 'text': 'Reran at %s bits. Verdicts in order of the tapes. %s' % (prec, ', '.join(verdicts))}
        if 'single' in q or 'float32' in q:
            return {'action': 'run', 'bits': 24, 'inputs': pairs or None, 'tapes': 'all'}
        if pairs:
            return {'action': 'run', 'bits': None, 'inputs': pairs, 'tapes': 'all'}
        verdict = (re.search(r'verdict\s+(.*)', user) or [None, 'see the report'])[1]
        return {'action': 'answer', 'text': 'From the report. ' + verdict}


# ----------------------------------------------------------------------------
# 1.  PROMPTS  (the contract)
# ----------------------------------------------------------------------------
TAPE_LANGUAGE = """TAPE LANGUAGE
A tape is a list of lines  name = expression.  One rounded operation per line is preferred.
Allowed operations: +  -  *  /  x*x  x**2  sqrt(x)  exp(x)  log(x) (natural)  log10(x)  log2(x)
  sin cos tan asin acos atan sinh cosh tanh abs expm1(x) log1p(x) pow(x, y)
Operands are input names, earlier line names, or decimal constants.  The last line must be named y.
Keep the user's order of operations exactly in tape 1.  Never simplify tape 1.  Never do arithmetic yourself.
Numbers for inputs are copied as the user wrote them, as strings."""

PROPOSE_SYSTEM = 'ROLE PROPOSE\n' + TAPE_LANGUAGE + """

You turn a user's description of a computation into tapes.  Reply with ONE JSON object and nothing else:
{"reading": "<one line, what you understood>",
 "inputs": {"<name>": "<value as written>", ...},
 "missing": ["<input names that have no value>"],
 "tapes": [{"name": "<short>", "lines": ["t1 = ...", "...", "y = ..."], "why": "<one line>"}, ...]}
Tape 1 is the user's formula exactly as written.  Tapes 2 and 3 (optional) are algebraically EQUAL ways to
compute the same value that avoid subtracting nearly equal numbers (conjugate trick, half angle, expm1, log1p,
factoring).  Only propose a rewrite you are sure is equal for all inputs in the domain.

EXAMPLE 1
USER TEXT  the root -b + sqrt(b^2 - c) of x^2 + 2bx + c = 0, b = 1000, c = 1
{"reading": "y = -b + sqrt(b^2 - c)", "inputs": {"b": "1000", "c": "1"}, "missing": [],
 "tapes": [{"name": "textbook", "lines": ["t1 = b*b", "t2 = t1 - c", "t3 = sqrt(t2)", "y = -b + t3"], "why": "as written"},
           {"name": "conjugate", "lines": ["t1 = b*b", "t2 = t1 - c", "t3 = sqrt(t2)", "t4 = b + t3", "y = -c / t4"],
            "why": "multiply top and bottom by b + sqrt(b^2 - c)"}]}

EXAMPLE 2
USER TEXT  (1 - cos x)/x^2 at x = 1e-4
{"reading": "y = (1 - cos(x)) / x^2", "inputs": {"x": "1e-4"}, "missing": [],
 "tapes": [{"name": "as written", "lines": ["t1 = cos(x)", "t2 = 1 - t1", "t3 = x*x", "y = t2 / t3"], "why": "as written"},
           {"name": "half angle", "lines": ["t1 = x/2", "t2 = sin(t1)", "t3 = t2*t2", "t4 = 2*t3", "t5 = x*x", "y = t4 / t5"],
            "why": "1 - cos x = 2 sin^2(x/2)"}]}

EXAMPLE 3
USER TEXT  Horner for 2x^3 - 3x + 1
{"reading": "y = ((2*x + 0)*x - 3)*x + 1 by Horner", "inputs": {}, "missing": ["x"],
 "tapes": [{"name": "Horner", "lines": ["t1 = 2*x", "t2 = t1*x", "t3 = t2 - 3", "t4 = t3*x", "y = t4 + 1"], "why": "as written"}]}
"""

FIX_PROMPT = """Your previous JSON had problems listed below.  Return the corrected JSON object only.
PROBLEMS
%s
PREVIOUS
%s"""

EXPLAIN_SYSTEM = 'ROLE EXPLAIN\n' + """You are a numerical analyst explaining a rounding error report to a student.
The report was produced by a reverse mode tape (Linnainmaa 1976, Bindel 2026).  Write four short paragraphs,
each starting with its title on the same line.
  What actually happened.   computed value, true value, actual error, and that the first order sum predicted it
  What the error might be.  ceiling, typical, kappa (what the problem forces) versus A (what the algorithm allows)
  Who is to blame and why.  the line with the biggest share, and the cancelling step that magnifies it
  What to do.               the best candidate from the comparison, or why none is needed
Rules.  Every number you write must be copied from the report.  At most three sentences per paragraph.
Plain everyday words.  No dashes, colons or semicolons inside sentences.  %s"""

AGENT_SYSTEM = 'ROLE AGENT\n' + """You help a student explore a rounding error report.  You have ONE tool, run, which reruns the
same tapes with new inputs or a new precision.  Reply with ONE JSON object only, either
  {"action": "run", "inputs": {"<name>": "<value>", ...} or null, "bits": 53 or 24 or 11 or null, "tapes": "all"}
  {"action": "answer", "text": "<plain words, numbers copied from the report, no dashes colons semicolons>"}
Use run when the question changes numbers or precision.  Otherwise answer from the report.  %s"""


# ----------------------------------------------------------------------------
# 2.  PROPOSE  (LLM)  and  3. VALIDATE  (engine)
# ----------------------------------------------------------------------------
def extract_json(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.M).strip()
    i, j = text.find('{'), text.rfind('}')
    if i < 0 or j < 0:
        raise ValueError('the model did not return JSON')
    return json.loads(text[i:j + 1])


def validate(proposal):
    """returns (readings, problems).  readings = list of (name, why, reading dict from rounderr)"""
    readings, problems = [], []
    inputs = proposal.get('inputs') or {}
    for k, t in enumerate(proposal.get('tapes') or [], 1):
        lines = t.get('lines') or []
        if not lines or not lines[-1].strip().startswith('y'):
            problems.append('tape %d must end with a line named y' % k)
            continue
        try:
            rd = rr.read_formula('\n'.join(lines), list(inputs))
            rd['typed'] = '\n'.join(lines)
        except Exception as err:
            problems.append('tape %d does not parse (%s)' % (k, err))
            continue
        missing = [v for v in rd['vars'] if v not in inputs]
        if missing:
            problems.append('tape %d uses %s but inputs has no value for it' % (k, ', '.join(missing)))
            continue
        for v in inputs:
            try:
                rr.parse_value(str(inputs[v]))
            except Exception:
                problems.append('input %s = %r is not a number' % (v, inputs[v]))
        readings.append((t.get('name', 'tape %d' % k), t.get('why', ''), rd))
    if not readings and not problems:
        problems.append('no tapes given')
    return readings, problems


def propose(llm, user_text, log):
    raw = llm.complete(PROPOSE_SYSTEM, 'USER TEXT  ' + user_text)
    log('LLM proposal received (%d chars)' % len(raw))
    proposal = extract_json(raw)
    readings, problems = validate(proposal)
    if problems:                                     # self refine, one round
        log('validation problems, asking the model to fix. ' + ' | '.join(problems))
        raw = llm.complete(PROPOSE_SYSTEM, FIX_PROMPT % ('\n'.join(problems), raw) + '\nUSER TEXT  ' + user_text)
        proposal = extract_json(raw)
        readings, problems = validate(proposal)
    return proposal, readings, problems


# ----------------------------------------------------------------------------
# 4.  VERIFY  (engine)   same math?  evaluate every tape in 60 digits, here and nearby
# ----------------------------------------------------------------------------
def ref_value(rd, values):
    tape = rr.Tape.from_reading(rd)
    rr.forward(tape, values, 53)
    return rr.reference(tape)


def perturbed(values, n=3, seed=1):
    rng = random.Random(seed)
    out = [values]
    for _ in range(n):
        s = {}
        for k, v in values.items():
            f = float(v)
            s[k] = Fraction(f * (1 + rng.uniform(-0.3, 0.3))) if f != 0 else Fraction(rng.uniform(-1, 1))
        out.append(s)
    return out


def verify(readings, values, user_formula_reading=None):
    """returns list of (ok, note) parallel to readings. Tape 1 is the reference, or the user's own formula if it parsed."""
    base = user_formula_reading or readings[0][2]
    out = []
    for name, why, rd in readings:
        bad, tested = [], 0
        for sample in perturbed(values):
            try:
                v0, vk = ref_value(base, sample), ref_value(rd, sample)
            except Exception:
                continue
            tested += 1
            scale = max(abs(v0), rr.hp(Fraction(1, 10 ** 300)))
            if abs(vk - v0) / scale > 1e-30:
                bad.append((sample, v0, vk))
        if tested == 0:
            out.append((False, 'could not evaluate at any sample'))
        elif bad:
            s, v0, vk = bad[0]
            out.append((False, 'wrong algebra, at %s it gives %s but the reference gives %s'
                        % (', '.join('%s = %s' % (k, rr.fnum(float(v))) for k, v in s.items()), rr.mpstr(vk), rr.mpstr(v0))))
        else:
            out.append((True, 'agrees with the reference to 30 digits at %d sample points' % tested))
    return out


# ----------------------------------------------------------------------------
# 5.  ANALYZE  (engine)   and   6. EXPLAIN  (LLM + grounding check)
# ----------------------------------------------------------------------------
def analyze_all(readings, values, bits):
    results = []
    for name, why, rd in readings:
        results.append(rr.analyze(rd, values, bits, label=name + ('  (' + why + ')' if why else '')))
    return results


def report_text(results):
    txt = '\n\n'.join(rr.text_report(R) for R in results)
    if len(results) > 1:
        txt += rr.comparison_text(results)
    return txt


NUM_RE = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])', re.I)


def numbers_in(text):
    out = []
    for m in NUM_RE.finditer(text.replace('×10^', 'e').replace('x10^', 'e')):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def grounding_check(narrative, report):
    """every number in the narrative must be within 5 percent of some number in the report"""
    have = numbers_in(report)
    loose = []
    for x in numbers_in(narrative):
        if x in (0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 53.0, 24.0, 60.0, 100.0):
            continue
        if not any(abs(x - h) <= 0.05 * max(abs(x), abs(h)) for h in have):
            loose.append(x)
    return loose


def explain(llm, results, user_text, lang='en', log=print):
    rep = report_text(results)
    extra = 'Answer in Chinese.' if lang == 'zh' else ''
    narrative = llm.complete(EXPLAIN_SYSTEM % extra, 'USER TEXT  ' + user_text + '\n\nREPORT\n' + rep)
    loose = grounding_check(narrative, rep)
    if loose:
        log('GROUNDING CHECK  these numbers in the explanation are not in the report  ' + ', '.join(rr.fnum(x) for x in loose))
    else:
        log('GROUNDING CHECK  every number in the explanation appears in the report')
    return narrative, loose


# ----------------------------------------------------------------------------
# 7.  FOLLOW UP  (agent loop with one tool)
# ----------------------------------------------------------------------------
class Session:
    def __init__(self, llm, user_text, readings, values, bits, lang='en', log=print):
        self.llm, self.user_text, self.readings = llm, user_text, readings
        self.values, self.bits, self.lang, self.log = values, bits, lang, log
        self.results = analyze_all(readings, values, bits)
        self.history = []                      # (question, answer)
        self.narrative = ''

    def ask(self, question, max_tool_calls=3):
        extra = 'Answer in Chinese.' if self.lang == 'zh' else ''
        context = 'USER TEXT  ' + self.user_text + '\n\nREPORT\n' + report_text(self.results)
        for q, a in self.history[-4:]:
            context += '\n\nEARLIER QUESTION  %s\nEARLIER ANSWER  %s' % (q, a)
        for _ in range(max_tool_calls):
            raw = self.llm.complete(AGENT_SYSTEM % extra, context + '\n\nQUESTION  ' + question)
            try:
                act = extract_json(raw)
            except ValueError:
                act = {'action': 'answer', 'text': raw}
            if act.get('action') != 'run':
                text = act.get('text', raw)
                loose = grounding_check(text, context)
                if loose:
                    text += '\n[grounding check, numbers not found in the report, %s]' % ', '.join(rr.fnum(x) for x in loose)
                self.history.append((question, text))
                return text
            # the one tool
            new_vals = dict(self.values)
            for k, v in (act.get('inputs') or {}).items():
                if k in new_vals:
                    new_vals[k] = rr.parse_value(str(v))
            bits = act.get('bits') or self.bits
            self.log('tool run   inputs %s   bits %s' % (', '.join('%s = %s' % (k, rr.fnum(float(v))) for k, v in new_vals.items()), bits))
            self.values, self.bits = new_vals, bits
            self.results = analyze_all(self.readings, new_vals, bits)
            context = 'USER TEXT  ' + self.user_text + '\n\nREPORT (after your run)\n' + report_text(self.results)
        text = 'I reran the engine. Here is the comparison.' + (rr.comparison_text(self.results) if len(self.results) > 1
                                                                 else '\n' + rr.text_report(self.results[0], show_tape=False))
        self.history.append((question, text))
        return text


# ----------------------------------------------------------------------------
# 8.  LaTeX  (engine report + the AI reading and narrative)
# ----------------------------------------------------------------------------
def latex_ai(session, proposal, checks, narrative, loose, path, pdf=False):
    tex = rr.latex_report(session.results, title='AI powered rounding error report')
    T = rr.tex_text
    block = [r'\section*{The AI layer}']
    block.append(r'\begin{rbox}[title={What the model read}]{askblue}')
    block.append('You typed ' + rr.verb(session.user_text) + r'\\')
    block.append('The model read it as ' + rr.verb(str(proposal.get('reading', ''))) + r'\\')
    block.append('Inputs ' + ', '.join('$%s = %s$' % (k, T(str(v))) for k, v in (proposal.get('inputs') or {}).items()))
    block.append(r'\end{rbox}')
    block.append(r'\begin{rbox}[title={Tapes proposed by the model and the engine verdict on their algebra}]{amber}')
    block.append(r'\begin{enumerate}')
    for (name, why, rd), (ok, note) in zip(session.readings_all, checks):
        block.append(r'\item \textbf{%s} (%s) \\ %s \\ %s %s' % (T(name), T(why), ' '.join(rr.verb(l) + r'\ ' for l in rd['typed'].splitlines()),
                                                                 r'\textcolor{chipgreen}{kept}' if ok else r'\textcolor{trapred}{dropped}', T(note)))
    block.append(r'\end{enumerate}')
    block.append(r'\end{rbox}')
    block.append(r'\begin{rbox}[title={The model explains the numbers}]{chipgreen}')
    for para in narrative.strip().split('\n\n'):
        block.append(T(para) + r'\par\medskip')
    block.append(r'Grounding check. ' + (r'every number above appears in the engine report.' if not loose
                                         else r'\textcolor{trapred}{numbers not in the report, %s}' % T(', '.join(rr.fnum(x) for x in loose))))
    block.append(r'\end{rbox}')
    if session.history:
        block.append(r'\begin{rbox}[title={Follow up questions}]{askblue}')
        for q, a in session.history:
            block.append(r'\textbf{Q} ' + T(q) + r'\\ \textbf{A} ' + T(a) + r'\par\medskip')
        block.append(r'\end{rbox}')
    tex = tex.replace(r'\end{document}', '\n'.join(block) + '\n' + r'\end{document}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    msg = 'wrote ' + path
    if pdf and shutil.which('pdflatex'):
        d = os.path.dirname(os.path.abspath(path)) or '.'
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', os.path.basename(path)],
                           cwd=d, capture_output=True, text=True)
        msg += '  and ' + path[:-4] + '.pdf' if r.returncode == 0 else '  (pdflatex failed, see the .log)'
    return msg


# ----------------------------------------------------------------------------
# 9.  the pipeline, command line, interactive loop, self test
# ----------------------------------------------------------------------------
def make_llm(provider, model=None, base_url=None):
    if provider == 'anthropic':
        return AnthropicLLM(model)
    if provider == 'openai':
        return OpenAICompatLLM(model, base_url)
    return MockLLM()


def pipeline(llm, user_text, bits=53, lang='en', log=print, extra_values=None):
    log('[1] PROPOSE   asking the %s model' % llm.name)
    proposal, readings, problems = propose(llm, user_text, log)
    if extra_values:
        proposal.setdefault('inputs', {}).update(extra_values)
        readings, problems = validate(proposal)
    log('    read as   ' + str(proposal.get('reading', '')))
    for k, (name, why, rd) in enumerate(readings, 1):
        log('    tape %d    %s   %s' % (k, name, ' | '.join(rd['typed'].splitlines())))
    if problems:
        log('    PROBLEMS  ' + ' | '.join(problems))
    if not readings:
        raise ValueError('no usable tape. ' + ' | '.join(problems))
    values = {k: rr.parse_value(str(v)) for k, v in (proposal.get('inputs') or {}).items()}
    # did the model transcribe the user's own formula correctly?  (only when the text itself parses)
    user_rd = None
    try:
        cand = re.split(r'\bwith\b|\bat\b|,|;', user_text, maxsplit=1)[0]
        tmp = rr.read_formula(cand, list(values))
        if all(v in values for v in tmp['vars']) and not re.search(r'[A-Za-z]{4,}', cand.replace('sqrt', '')):
            user_rd = tmp
    except Exception:
        user_rd = None
    log('[3] VERIFY    60 digit check of every tape against %s' % ('your own formula' if user_rd else 'tape 1'))
    checks = verify(readings, values, user_rd)
    kept = []
    for (name, why, rd), (ok, note) in zip(readings, checks):
        log('    %-8s %s   %s' % ('kept' if ok else 'DROPPED', name, note))
        if ok:
            kept.append((name, why, rd))
    if not kept:
        raise ValueError('every tape failed verification')
    log('[4] ANALYZE   forward pass, slips, backward sweep')
    session = Session(llm, user_text, kept, values, bits, lang, log)
    session.readings_all = readings
    log('[5] EXPLAIN   asking the model to put the numbers into words')
    narrative, loose = explain(llm, session.results, user_text, lang, log)
    session.narrative = narrative
    return proposal, checks, session, narrative, loose


def selftest():
    print('rounderr_ai self test (mock model)')
    quiet = lambda *a, **k: None
    ok = 0
    # 1 a wrong rewrite must be dropped by verification
    proposal, checks, S, narrative, loose = pipeline(MockLLM(), 'wrong x = 0.5', log=quiet)
    good = checks[0][0] and not checks[1][0] and len(S.results) == 1
    print('  %s  wrong algebra is dropped by the 60 digit check' % ('PASS' if good else 'FAIL'))
    ok += good
    # 2 the quadratic gives two tapes, both kept, and the stable one wins
    proposal, checks, S, narrative, loose = pipeline(MockLLM(), 'positive root of x^2 + 2bx + c = 0 with b = 1000, c = 1', log=quiet)
    good = all(c[0] for c in checks) and S.results[0].amp > 1e5 and S.results[1].amp < 5
    print('  %s  quadratic, two tapes kept, A = %s and %s' % ('PASS' if good else 'FAIL', rr.fnum(S.results[0].amp), rr.fnum(S.results[1].amp)))
    ok += good
    # 3 the narrative is grounded
    good = not loose
    print('  %s  narrative numbers all found in the report' % ('PASS' if good else 'FAIL'))
    ok += good
    # 4 grounding check catches an invented number
    good = grounding_check('the error is 7.77e-5 which is 12345 u', report_text(S.results)) != []
    print('  %s  grounding check catches an invented number' % ('PASS' if good else 'FAIL'))
    ok += good
    # 5 agent tool call changes the precision
    S.ask('what happens in single precision')
    good = S.bits == 24 and S.results[0].bits == 24
    print('  %s  follow up "single precision" reruns the engine at 24 bits' % ('PASS' if good else 'FAIL'))
    ok += good
    # 6 agent tool call changes an input
    S.ask('and with b = 1e6')
    good = float(S.values['b']) == 1e6
    print('  %s  follow up "b = 1e6" reruns with the new input (A = %s)' % ('PASS' if good else 'FAIL', rr.fnum(S.results[0].amp)))
    ok += good
    # 7 free formula text goes through the generic path and matches the user's own formula
    proposal, checks, S, narrative, loose = pipeline(MockLLM(), 'log(1 + x) at x = 1e-10', log=quiet)
    good = checks[0][0] and S.results[0].amp > 1e9
    print('  %s  plain formula text, transcription verified against the user formula' % ('PASS' if good else 'FAIL'))
    ok += good
    print('%d of 7 passed' % ok)


def main():
    ap = argparse.ArgumentParser(description='AI powered reverse mode rounding error interpreter')
    ap.add_argument('text', nargs='?', help='what to analyze, in your own words, a formula, or LaTeX')
    ap.add_argument('values', nargs='*', help='optional name=value pairs that override the model')
    ap.add_argument('--provider', choices=['anthropic', 'openai', 'mock'], default='anthropic')
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--bits', type=int, default=53)
    ap.add_argument('--lang', choices=['en', 'zh'], default='en')
    ap.add_argument('--tex', metavar='FILE')
    ap.add_argument('--pdf', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    try:
        llm = make_llm(args.provider, args.model, args.base_url)
    except ImportError:
        print('pip install anthropic   (or use --provider openai / --provider mock)')
        return
    if args.provider == 'anthropic' and not os.environ.get('ANTHROPIC_API_KEY'):
        print('set ANTHROPIC_API_KEY, or use --provider mock to see the pipeline offline')
        return
    text = args.text or input('describe the computation > ')
    extra = {}
    for kv in args.values:
        k, v = kv.split('=', 1)
        extra[k.strip()] = v.strip()
    proposal, checks, S, narrative, loose = pipeline(llm, text, args.bits, args.lang, extra_values=extra)
    print()
    print(report_text(S.results))
    print()
    print('THE MODEL EXPLAINS')
    print(narrative)
    print()
    while True:
        try:
            q = input('follow up (Enter to stop) > ').strip()
        except EOFError:
            q = ''
        if not q:
            break
        print(S.ask(q))
    if args.tex:
        print(latex_ai(S, proposal, checks, narrative, loose, args.tex, pdf=args.pdf))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
    except (ValueError, json.JSONDecodeError) as err:
        print('rounderr_ai could not finish. %s' % err)
        sys.exit(1)
