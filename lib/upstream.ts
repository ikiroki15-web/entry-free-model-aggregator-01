// Dialling one provider in its own wire.
//
// The caller builds the upstream request body for the (client wire,
// provider wire) pair; this module sends it to the provider's route, with
// the provider's own auth header, and hands back either the JSON answer or
// the whole SSE text. A refusal (402, 429, 5xx, a bad key, an unknown model)
// comes back as ok: false with the status, so the caller can fail over to
// the next candidate instead of answering with the refusal.
//
// One call at a time per provider: the client sends requests sequentially,
// and this module never runs two provider calls in parallel, so a provider's
// single in-flight slot is never contended by this entry itself.

import { ANTHROPIC_VERSION, authHeaders, isRecord, joinBase, type Json, type Probe } from './brief.ts';

/** How long one provider call may take; the client waits 30 s per request. */
const CALL_TIMEOUT_MS = 25000;

export interface UpstreamResult {
  ok: boolean;
  status: number;
  /** The JSON answer, when the call succeeded and was not streamed. */
  json: Json | null;
  /** The whole SSE text, when the call succeeded and was streamed. */
  streamText: string | null;
  /** The refusal's body, for the log, when the call did not succeed. */
  errorText: string;
}

function failed(status: number, errorText: string): UpstreamResult {
  return { ok: false, status, json: null, streamText: null, errorText };
}

/**
 * POST one request body to a provider. `path` is the provider route below
 * its base (for example `/v1/chat/completions`, or for Gemini
 * `/v1beta/models/<model>:generateContent`); `stream` reads the SSE body
 * whole instead of parsing JSON.
 */
export async function dialProvider(probe: Probe, envUrl: string, path: string, body: Json, stream: boolean): Promise<UpstreamResult> {
  const url = `${joinBase(envUrl, probe.provider.base)}${path}`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (probe.provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return failed(0, `dial ${url} failed: ${(err as Error).message}`);
  }
  let text: string;
  try {
    text = await res.text();
  } catch (err) {
    return failed(0, `reading ${url} failed: ${(err as Error).message}`);
  }
  if (!res.ok) return failed(res.status, text.slice(0, 500));
  if (stream) return { ok: true, status: res.status, json: null, streamText: text, errorText: '' };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return failed(502, `the provider answered ${url} with a body that is not JSON`);
  }
  if (!isRecord(parsed)) return failed(502, `the provider answered ${url} with a JSON value that is not one object`);
  return { ok: true, status: res.status, json: parsed, streamText: null, errorText: '' };
}

/** The upstream route for a provider wire, given the model and streaming. */
export function upstreamPath(wire: Probe['provider']['wire'], model: string, stream: boolean): string {
  if (wire === 'openai') return '/v1/chat/completions';
  if (wire === 'anthropic') return '/v1/messages';
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const suffix = stream ? '?alt=sse' : '';
  return `/v1beta/models/${encodeURIComponent(model)}:${verb}${suffix}`;
}

/** Whether a refusal is worth failing over from (any non-2xx already is). */
export function isRefusal(result: UpstreamResult): boolean {
  return !result.ok;
}
