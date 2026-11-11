# rounderr

A reverse mode rounding error machine. Type a formula, and it shows where floating point
error comes from, which step is to blame, whether the algorithm is stable, and how to rewrite it.

**Live: https://erics1129.github.io/rounderr/**

The live page runs the real Python engine inside your browser (Pyodide, CPython as WebAssembly).
There is no server. Nothing you type leaves the page.

By Eric Shi, CS 4999 with Dr. David Bindel. The math is Linnainmaa's reverse mode error analysis
in Bindel's notation. The full write up is in [rounderr_design.pdf](rounderr_design.pdf).

## What you see

Nothing waits for a button. Every keystroke reruns the analysis, and every analysis is followed
by three experiments on their own.

| Picture | What it shows |
|---|---|
| Verdict and digits | Stable or unstable, measured error, ceiling A, condition number κ, and how many digits you can trust |
| Error ladder | One log axis in units of u. What the problem forces (κ), what the algorithm allows (A), the limit, and the error actually measured. One row per rewrite |
| Tape as a graph | The computation graph, animated. Values flow forward, λ flows back, blame lands on the guilty step |
| Who is to blame | Share of the error ceiling carried by each rounded step |
| Every precision | The same run at every precision from 8 to 53 bits. Click a point to rerun there |
| Random trials | Thousands of random inputs around yours as a cloud, κ against A, with the stability boundary |
| Input landscape | A/κ along one input (a curve) or across two (a map). Click anywhere to move your numbers there |

Plus the original detail tabs: Tape, Trials, Matrix (L, B, w, λ), Bean, Data (CSV and LaTeX export) and
Developer (every stage with its trace and its source code).

Formulas with no numbers still run. The machine picks numbers that give a finite result and marks
them with a dashed box until you change them.

## Run it on your own computer

The hosted page covers everything except two things that need native programs: the
[Bean](https://github.com/Athena-Types/Bean) type checker and the Claude or local model readers.

```bash
pip install numpy mpmath
python rounderr_app.py
```

That opens http://127.0.0.1:8765 with the same window, backed by native Python.

| Option | Meaning |
|---|---|
| `--bean path/to/bean.exe` | use Bean for the static backward error bound |
| `ANTHROPIC_API_KEY=...` | switch on the Claude reader, which reads plain words (`pip install anthropic`) |
| `OPENAI_BASE_URL=...` | a local model behind an OpenAI style endpoint |
| `--selftest` | 8 checks of the app, including the fast sweep against the full engine |
| `--trace "formula with x = 1"` | print the developer trace in the terminal |
| `--snapshot` | rewrite `snapshot.json`, the saved run the hosted page paints before Python is up |

The command line tools still work on their own: `python rounderr.py`, `python rounderr_ai.py`,
`python rounderr_bean.py`.

## How it is built

```
index.html, rounderr.css      the window
rounderr_ui.js                auto run, tabs, events, and api(route, body)
rounderr_charts.js            every picture, hand drawn SVG, HTML and canvas, no chart library
        |
        |  api() has two engines behind it, same routes, same JSON
        |
        +--  http     rounderr_app.py serving on localhost          native Python
        +--  worker   engine-worker.js, Pyodide in a web worker     the same four .py files

rounderr.py        the engine. Reader, tape, forward pass, backward sweep, numbers, rewrites
rounderr_ai.py     the model layer. Proposal, 60 digit verification, grounded explanation
rounderr_bean.py   the Bean bridge
rounderr_app.py    hash table, fast numpy sweep, trials, precision sweep, landscape, server, bridge()
```

Why it feels fast

* One analysis takes 1 to 3 ms in native Python and about 10 ms in the browser.
* A newer keystroke makes older answers stale. They are dropped by run id, never drawn.
* The experiments run after the analysis is on screen, in the order cheapest first.
* The hosted page paints a saved run at once, then swaps to live answers when Python is up
  (about 3 seconds on a warm cache). numpy loads in the background after the first answer.

## Tests

```bash
python rounderr_app.py --selftest
python rounderr.py --selftest
python rounderr_ai.py --selftest
```
