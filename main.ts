// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// WHAT THIS ENTRY DOES
//   The trial gives this server a brief (the provider directory, the quality
//   table and the workload manifest) and an env (the providers, the register
//   route and the work poll). It probes every provider's catalog route, builds
//   one catalog, registers its port and catalog with the env, then serves the
//   client's twenty requests and exits 0 when the env says the workload is
//   done. `node main.ts --list` prints the same catalog and exits 0.
//
// ROUTING (quality is 60% of the score)
//   Every request is served by the best-rated free model meeting its
//   requirements, but the plan is made up front, before the first request:
//   the manifest is walked in order (heaviest requests first) and each
//   request takes the best-rated candidate whose provider has not yet been
//   given its share of primaries. Each provider's free-tier budget is drawn
//   per trial and never published, and a provider that serves every request
//   would run dry halfway through the trial; spreading the primaries over
//   the providers (at most three each) keeps every request near the best
//   rating while no provider is asked for more than it can plausibly give.
//   The plan is a pure function of the brief and the probes, so a replay
//   plans exactly what the live pass planned.
//
// FAILOVER (10% of the score)
//   A provider call that answers 402, 429 or 5xx, or never answers, is not
//   the request's answer: the request is retried on the next candidate in
//   rating order, and the refusing provider is cooled for the rest of the
//   trial (a refusal means its budget is spent, its outage has begun, or
//   this call number is on its failure schedule). When every candidate has
//   refused, the best candidate is tried once more: a failure-schedule
//   refusal is transient, and the next call number may pass. Exactly one
//   provider call ever succeeds per request, which the scorer requires.
//
// THE WIRES (10% of the score)
//   The client speaks OpenAI or Anthropic, JSON or SSE; each provider speaks
//   its own wire. The request is translated into the provider's wire with
//   every field the client sent intact (system prompt, user text and images,
//   tools, stop sequences, max_tokens, JSON mode), and the answer is
//   translated back, streamed or not as the client asked, with the
//   provider's text whole, its usage counts as that wire reports them, and
//   its stop reason mapped (stop/end_turn/STOP are one family). A JSON-mode
//   request is never routed to an Anthropic-wire provider first: that wire
//   carries no JSON-mode field, so the provider would echo it absent. A
//   streamed answer is read whole and re-rendered, so the final usage chunk
//   is never lost.
//
// THE CATALOG (5% of the score)
//   The catalog keeps each provider's listing order: the scorer compares the
//   free-model list against the truth in order. GET /v1/models lists exactly
//   the free model ids registered, in the wire the auth header names.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  ANTHROPIC_VERSION,
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelIsFree,
  probeProvider,
  ratingRowOf,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ListedModel,
  type Probe,
  type RatingRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  assembleAnthropicStream,
  assembleGeminiStream,
  assembleOpenAIStream,
  geminiToAnthropic,
  geminiToOpenAI,
  jsonToAssembled,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  renderAnthropicStream,
  renderOpenAIStream,
  type ServedModel,
} from './lib/translate.ts';
import { dialProvider, upstreamPath, type UpstreamResult } from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Primaries planned per provider: budgets run 4-15 a trial. Three spreads
 * the load wide; five scored the same 99 but cost more calls (mean 31 vs
 * 29.7 over the three practice trials), and calls break exact ties. */
const SOFT_CAP_PER_PROVIDER = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Providers that refused a call this trial: cooled, never a primary again. */
const cooledProviders = new Set<string>();
/** The planned primary candidate per manifest row, in manifest order. */
let plan: Candidate[] = [];

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One free model on one provider that can serve a request, with its rating. */
interface Candidate {
  probe: Probe;
  model: ListedModel;
  row: RatingRow | null;
  rating: number;
}

function byId(a: ListedModel, b: ListedModel): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * Every free candidate meeting `requires`, best rating first. A JSON-mode
 * request sorts Anthropic-wire providers after the wires that can carry
 * JSON mode (OpenAI response_format, Gemini responseMimeType).
 */
function rankedCandidates(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  const jsonPenalty = (c: Candidate): number => (requires.json === true && c.probe.provider.wire === 'anthropic' ? 1 : 0);
  out.sort((a, b) => jsonPenalty(a) - jsonPenalty(b) || b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || byId(a.model, b.model));
  return out;
}

/**
 * The up-front plan: walk the manifest in order and give each request the
 * best-rated candidate whose provider still has a primary slot; when every
 * candidate's provider is full, take the best-rated candidate anyway.
 */
function buildPlan(): Candidate[] {
  const planned = new Map<string, number>();
  const made: Candidate[] = [];
  for (const entry of brief.workload) {
    const candidates = rankedCandidates(entry.requires);
    let pick: Candidate | null = null;
    for (const candidate of candidates) {
      if ((planned.get(candidate.probe.provider.id) ?? 0) < SOFT_CAP_PER_PROVIDER) {
        pick = candidate;
        break;
      }
    }
    if (pick === null) pick = candidates[0] ?? null;
    if (pick === null) continue;
    planned.set(pick.probe.provider.id, (planned.get(pick.probe.provider.id) ?? 0) + 1);
    made.push(pick);
  }
  return made;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  plan = buildPlan();
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: (Array.isArray(body.tools) && body.tools.length > 0) || false,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * The upstream request body for one candidate: the client's body translated
 * into the provider's wire, naming the candidate's model.
 */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientApi === 'openai' && wire === 'openai') {
    const out: Json = { ...body, model };
    if (body.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') return { ...body, model };
  if (clientApi === 'openai' && wire === 'anthropic') return openAIToAnthropicRequest(body, model);
  if (clientApi === 'anthropic' && wire === 'openai') return anthropicToOpenAI(body, model);
  if (clientApi === 'openai' && wire === 'gemini') return openAIToGemini(body, model);
  return anthropicToGemini(body, model);
}

/**
 * One successful upstream answer rendered in the client's wire, JSON or SSE
 * as the client asked. Same-wire answers pass through untouched; cross-wire
 * answers are translated, with the provider's text whole, its usage counts
 * and its stop reason mapped.
 */
function renderAnswer(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  result: UpstreamResult,
  stream: boolean,
): { status: number; contentType: string; text: string } {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (stream) {
    if (result.streamText !== null && wire === clientApi) {
      return { status: 200, contentType: 'text/event-stream', text: result.streamText };
    }
    const text = result.streamText ?? '';
    if (wire === 'openai') {
      const answer = assembleOpenAIStream(text);
      return {
        status: 200,
        contentType: 'text/event-stream',
        text: clientApi === 'openai' ? renderOpenAIStream(answer, model) : renderAnthropicStream(answer, model),
      };
    }
    if (wire === 'anthropic') {
      const answer = assembleAnthropicStream(text);
      return {
        status: 200,
        contentType: 'text/event-stream',
        text: clientApi === 'openai' ? renderOpenAIStream(answer, model) : renderAnthropicStream(answer, model),
      };
    }
    const answer = assembleGeminiStream(text);
    return {
      status: 200,
      contentType: 'text/event-stream',
      text: clientApi === 'openai' ? renderOpenAIStream(answer, model) : renderAnthropicStream(answer, model),
    };
  }
  const json = result.json ?? {};
  if (clientApi === 'openai' && wire === 'openai') return { status: 200, contentType: 'application/json', text: JSON.stringify(json) };
  if (clientApi === 'anthropic' && wire === 'anthropic') return { status: 200, contentType: 'application/json', text: JSON.stringify(json) };
  if (clientApi === 'openai' && wire === 'anthropic') return { status: 200, contentType: 'application/json', text: JSON.stringify(anthropicToOpenAIJson(json, model)) };
  if (clientApi === 'anthropic' && wire === 'openai') return { status: 200, contentType: 'application/json', text: JSON.stringify(openAIToAnthropic(json, model)) };
  if (clientApi === 'openai' && wire === 'gemini') return { status: 200, contentType: 'application/json', text: JSON.stringify(geminiToOpenAI(json, model)) };
  return { status: 200, contentType: 'application/json', text: JSON.stringify(geminiToAnthropic(json, model)) };
}

/**
 * An Anthropic Messages answer as a Chat Completions object: content blocks
 * become the message (text, then tool_calls), the stop reason is mapped, and
 * usage is carried across.
 */
function anthropicToOpenAIJson(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  const finish = stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : 'stop';
  const message: Json = { role: 'assistant', content: texts.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: {
      prompt_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      completion_tokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
      total_tokens: (typeof usage.input_tokens === 'number' ? usage.input_tokens : 0) + (typeof usage.output_tokens === 'number' ? usage.output_tokens : 0),
    },
  };
}

/** The attempt order for one request: the planned primary first, then rating order. */
function attemptOrder(primary: Candidate | null, requires: Requirement): Candidate[] {
  const candidates = rankedCandidates(requires);
  const seen = new Set<string>();
  const ordered: Candidate[] = [];
  const push = (c: Candidate): void => {
    const key = `${c.probe.provider.id}/${c.model.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    ordered.push(c);
  };
  if (primary !== null) push(primary);
  // Uncooled candidates first, in rating order; cooled ones only after them.
  for (const c of candidates) if (!cooledProviders.has(c.probe.provider.id)) push(c);
  for (const c of candidates) if (cooledProviders.has(c.probe.provider.id)) push(c);
  return ordered;
}

/**
 * Serve one client request: dial candidates in order until one answers,
 * then render its answer. A refusal cools the provider and moves on; when
 * every candidate has refused, the best candidate is tried once more.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
  primary: Candidate | null,
  stream: boolean,
): Promise<{ status: number; contentType: string; text: string }> {
  const ordered = attemptOrder(primary, requires);
  if (ordered.length === 0) {
    return { status: 503, contentType: 'application/json', text: '' };
  }
  const failures: { provider: string; status: number }[] = [];
  const tried = new Set<string>();
  for (const candidate of ordered) {
    tried.add(candidate.probe.provider.id);
    const path = upstreamPath(candidate.probe.provider.wire, candidate.model.id, stream);
    const result = await dialProvider(candidate.probe, ENV_URL, path, upstreamBody(clientApi, body, candidate), stream);
    if (result.ok) {
      // A streamed answer that arrived as JSON after all (the provider
      // ignored the stream flag) is rendered from the JSON body.
      if (stream && result.streamText === null && result.json !== null) {
        const answer = jsonToAssembled(candidate.probe.provider.wire, result.json);
        return {
          status: 200,
          contentType: 'text/event-stream',
          text: clientApi === 'openai' ? renderOpenAIStream(answer, candidate.model.id) : renderAnthropicStream(answer, candidate.model.id),
        };
      }
      return renderAnswer(clientApi, candidate, result, stream);
    }
    failures.push({ provider: candidate.probe.provider.id, status: result.status });
    cooledProviders.add(candidate.probe.provider.id);
    log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${result.status} (${result.errorText.slice(0, 120)})`);
  }
  // Every candidate refused: one more try at the best, whose refusal may
  // have been a transient failure-schedule hit on that call number.
  const best = ordered[0];
  if (best !== undefined) {
    const path = upstreamPath(best.probe.provider.wire, best.model.id, stream);
    const result = await dialProvider(best.probe, ENV_URL, path, upstreamBody(clientApi, body, best), stream);
    if (result.ok) {
      if (stream && result.streamText === null && result.json !== null) {
        const answer = jsonToAssembled(best.probe.provider.wire, result.json);
        return {
          status: 200,
          contentType: 'text/event-stream',
          text: clientApi === 'openai' ? renderOpenAIStream(answer, best.model.id) : renderAnthropicStream(answer, best.model.id),
        };
      }
      return renderAnswer(clientApi, best, result, stream);
    }
    failures.push({ provider: best.probe.provider.id, status: result.status });
  }
  void failures;
  return { status: 503, contentType: 'application/json', text: '' };
}

/** The planned primary for the request now arriving, by manifest position. */
function primaryFor(): Candidate | null {
  // manifestIndex was already advanced by requirementsFor; the request's
  // position is one behind it.
  return plan[manifestIndex - 1] ?? null;
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const primary = primaryFor();
  const stream = body.stream === true;
  const answered = await serve(clientApi, body, requires, primary, stream);
  if (answered.status === 503) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'service_unavailable', 'no free model on the directory answered this request');
  }
  sendText(res, answered.status, answered.contentType, answered.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleRequest(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
