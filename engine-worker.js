/* engine-worker.js   the browser build of the rounderr machine
   ============================================================
   No server. This worker starts Pyodide (CPython compiled to WebAssembly), drops the four
   Python files into its file system, imports rounderr_app and calls rounderr_app.bridge()
   for every request the page sends. The page gets the same JSON the local server returns.

   Start up is staged so the first answer comes early:
     1  the Python runtime                       analysis cannot start before this
     2  mpmath and the four engine files         analysis is ready
     3  numpy, in the background                 random trials and the input landscape are ready
*/
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.29.5/full/';
const FILES = ['rounderr.py', 'rounderr_ai.py', 'rounderr_bean.py', 'rounderr_app.py'];
const NEEDS_NUMPY = new Set(['trials', 'landscape']);
const t0 = performance.now();
const boot = (stage, extra) => postMessage(Object.assign({type: 'boot', stage, ms: performance.now() - t0}, extra || {}));

let pyodide = null, bridge = null;

const ready = (async () => {
  boot('runtime');
  const sources = Promise.all(FILES.map(f => fetch(f).then(r => {
    if (!r.ok) throw new Error('could not fetch ' + f);
    return r.text();
  })));
  importScripts(PYODIDE + 'pyodide.js');
  pyodide = await loadPyodide({indexURL: PYODIDE});
  boot('packages');
  await pyodide.loadPackage('mpmath');
  const text = await sources;
  pyodide.FS.mkdirTree('/app');
  FILES.forEach((f, i) => pyodide.FS.writeFile('/app/' + f, text[i]));
  boot('engine');
  pyodide.runPython("import sys, os\nsys.path.insert(0, '/app')\nos.chdir('/app')\nimport rounderr_app");
  const app = pyodide.globals.get('rounderr_app');
  bridge = app.bridge;
  // one throwaway run so the first real answer does not pay for cold imports and caches
  bridge('analyze', JSON.stringify({text: '-b + sqrt(b^2 - c)', mode: 'rule', values: {b: '1000', c: '1'}}));
  pyodide.runPython('rounderr_app._BROWSER.clear()');
  postMessage({type: 'ready', ms: performance.now() - t0});
})();

const numpy = ready.then(async () => {
  await pyodide.loadPackage('numpy');
  pyodide.runPython('import numpy\nrounderr_app.use_numpy(numpy)');
  postMessage({type: 'numpy', ms: performance.now() - t0});
});

ready.catch(err => postMessage({type: 'fail', error: String(err && err.message || err)}));
numpy.catch(err => postMessage({type: 'fail', error: String(err && err.message || err)}));

onmessage = async e => {
  const {id, route, body} = e.data;
  let json;
  try {
    await ready;
    if (NEEDS_NUMPY.has(route)) await numpy;
    const t = performance.now();
    json = bridge(route, JSON.stringify(body || {}));
    postMessage({type: 'reply', id, json, ms: performance.now() - t});
  } catch (err) {
    postMessage({type: 'reply', id, json: JSON.stringify({error: 'The engine in the browser failed. ' + String(err && err.message || err)}), ms: 0});
  }
};
