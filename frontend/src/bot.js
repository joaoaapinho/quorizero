// Web Worker, so the page never freezes. Runs model/model.onnx with onnxruntime-web,
// or without a model, search over a shortest-path heuristic.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs';
import { N_PLANES, observe, orient } from './game.js';
import { chooseMove, heuristicEvaluate, softmaxOverLegal } from './search.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

let session = null;
// The win bar uses a 200-sim search, not the raw network value, which over-rates the
// first player in the opening (58% vs ~48% in self-play). Costs ~1 s.
const ODDS_SIMS = 200;

async function netEvaluate(state, legal) {
  const planes = new ort.Tensor('float32', observe(state), [1, 9, 9, N_PLANES]);
  const out = await session.run({ planes });
  const logits = orient(out.logits.data, state.toMove); // canonical -> board order
  return { priors: softmaxOverLegal(logits, legal), value: out.value.data[0] };
}

const ready = ort.InferenceSession.create(new URL('../model/model.onnx', import.meta.url).href)
  .then((s) => (session = s))
  .catch(() => null)
  .then(() => self.postMessage({ type: 'ready', model: session !== null }));

// One request at a time: a search and an evaluation never share the session.
async function handle(data) {
  await ready;
  const evaluate = session ? netEvaluate : heuristicEvaluate;
  if (data.type === 'evaluate') {
    const { value } = await chooseMove(data.state, { sims: ODDS_SIMS }, evaluate);
    self.postMessage({ type: 'value', id: data.id, ply: data.state.ply, value });
    return;
  }
  const result = await chooseMove(data.state, data.options, evaluate);
  self.postMessage({ type: 'move', id: data.id, ...result });
}

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(() => handle(data)).catch((error) => console.error(error));
};
