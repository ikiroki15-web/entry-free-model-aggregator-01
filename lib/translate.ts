// The three wires: OpenAI Chat Completions, Anthropic Messages, Gemini
// generateContent. This module translates requests and answers between them,
// so any provider can serve either kind of client.
//
// The translations are written to keep the provider's echo faithful to the
// client request: the simulated providers answer with an echo of what they
// parsed (system text, user text, tool names, max_tokens, stop sequences,
// image bytes, JSON mode), and the client compares that echo with what it
// sent. So every field the client sent must arrive at the provider intact:
// system prompts joined with "\n", the last user message's text and image,
// tool definitions by name, the output cap, the stop sequences, and JSON mode
// (OpenAI response_format, or Gemini responseMimeType; the Anthropic wire
// carries no JSON-mode field, so a JSON-mode request is never routed to an
// Anthropic-wire provider first).
//
// Streaming answers are read whole and re-rendered in the client's wire:
// the answers are short, and rendering from the assembled answer guarantees
// the event order, the whole text, the usage and the stop reason, where
// passing raw frames through could drop the final usage chunk (Gemini only
// reports usage on its last frame; OpenAI only when stream_options asks).
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so live and replay render the same bytes.

import { isRecord, type Json } from './brief.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** Fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** Fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in Gemini's shape. */
export function geminiError(code: number, status: string, message: string): Json {
  return { error: { code, message, status } };
}

/** The text of a string or of a block array's text blocks, joined with "\n". */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  return openAIStopToAnthropic(geminiStopToOpenAI(reason));
}

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Anthropic <-> OpenAI (requests)
// ---------------------------------------------------------------------------

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic content block as an OpenAI content part, or null. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block, or null. */
function openAIPartToBlock(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool messages become tool_result blocks, assistant tool_calls
 * become tool_use blocks, and the output cap, temperature, top_p, stop
 * sequences, tools and streaming pass through. JSON mode has no Anthropic
 * spelling, so response_format is dropped here on purpose: callers never
 * route a JSON-mode request to an Anthropic-wire provider first.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: text and image parts become Anthropic blocks.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((part) => {
        const block = openAIPartToBlock(part);
        return block === null ? [] : [block];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// OpenAI / Anthropic <-> Gemini (requests)
// ---------------------------------------------------------------------------

/** The mime type and data of a data URL, or null when it is not one. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const match = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mime: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI message as Gemini content parts, with the role they carry. */
function openAIMessageParts(message: Json): { role: 'user' | 'model'; parts: Json[] } | null {
  const role = message.role;
  if (role === 'system') return null;
  const parts: Json[] = [];
  if (role === 'tool') {
    parts.push({
      functionResponse: {
        name: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
        response: { result: typeof message.content === 'string' ? message.content : textOf(message.content) },
      },
    });
    return { role: 'user', parts };
  }
  if (role === 'assistant') {
    const text = textOf(message.content);
    if (text !== '') parts.push({ text });
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
    }
    return { role: 'model', parts };
  }
  // User message.
  if (typeof message.content === 'string') {
    if (message.content !== '') parts.push({ text: message.content });
  } else if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const data = dataUrlParts(part.image_url.url);
        if (data !== null) parts.push({ inlineData: { mimeType: data.mime, data: data.data } });
        else parts.push({ fileData: { fileUri: part.image_url.url, mimeType: 'image/png' } });
      }
    }
  }
  return { role: 'user', parts };
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: the
 * system messages become systemInstruction, user and assistant turns become
 * contents, tool definitions become functionDeclarations, and the output cap,
 * temperature, top_p, stop sequences and JSON mode (responseMimeType) pass
 * through generationConfig.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const converted = openAIMessageParts(raw);
    if (converted !== null && converted.parts.length > 0) contents.push({ role: converted.role, parts: converted.parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s) => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request, via OpenAI. */
export function anthropicToGemini(body: Json, _model: string): Json {
  return openAIToGemini(anthropicToOpenAI(body, 'gemini'), _model);
}

// ---------------------------------------------------------------------------
// Answers: JSON
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const message: Json = { role: 'assistant', content: texts.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: geminiStopToOpenAI(candidate.finishReason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object, via OpenAI. */
export function geminiToAnthropic(body: Json, model: string): Json {
  return openAIToAnthropic(geminiToOpenAI(body, model), model);
}

// ---------------------------------------------------------------------------
// Answers: streaming. Each reader consumes a whole upstream stream into one
// assembled answer; each renderer writes that answer in the client's wire.
// ---------------------------------------------------------------------------

export interface ToolCallDraft {
  id: string;
  name: string;
  arguments: string;
}

export interface AssembledAnswer {
  text: string;
  toolCalls: ToolCallDraft[];
  /** OpenAI-style finish_reason. */
  finish: string;
  inputTokens: number;
  outputTokens: number;
}

/** Read a ReadableStream to a string. */
export async function readStreamText(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let out = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      out += decoder.decode(value, { stream: true });
    }
    out += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return out;
}

/** Split SSE text into its frames: {event, data} pairs. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI SSE stream as one assembled answer. */
export function assembleOpenAIStream(text: string): AssembledAnswer {
  let out = '';
  const calls = new Map<number, ToolCallDraft>();
  let finish = 'stop';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    const deltaCalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of deltaCalls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      let draft = calls.get(index);
      if (draft === undefined) {
        draft = { id: '', name: '', arguments: '' };
        calls.set(index, draft);
      }
      if (typeof call.id === 'string' && call.id !== '') draft.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') draft.name = fn.name;
      if (typeof fn.arguments === 'string') draft.arguments += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { text: out, toolCalls: [...calls.values()], finish, inputTokens, outputTokens };
}

/** An Anthropic SSE stream as one assembled answer. */
export function assembleAnthropicStream(text: string): AssembledAnswer {
  let out = '';
  const calls = new Map<number, ToolCallDraft>();
  let finish = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    switch (frame.event) {
      case 'message_start':
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        break;
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
        else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
          const draft = calls.get(index);
          if (draft !== undefined) draft.arguments += delta.partial_json;
        }
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          calls.set(index, {
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            arguments: '',
          });
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') finish = delta.stop_reason;
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  // Anthropic stop reasons are already in the client's family; keep them, and
  // map them to the OpenAI family for the shared renderer input.
  const openAIFinish = finish === 'max_tokens' ? 'length' : finish === 'tool_use' ? 'tool_calls' : finish === 'refusal' ? 'content_filter' : 'stop';
  return { text: out, toolCalls: [...calls.values()], finish: openAIFinish, inputTokens, outputTokens };
}

/** A Gemini SSE stream as one assembled answer. */
export function assembleGeminiStream(text: string): AssembledAnswer {
  let out = '';
  const calls: ToolCallDraft[] = [];
  let finish = 'stop';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(text)) {
    const data = parseJsonObject(frame.data.trim());
    if (data === null) continue;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    if (typeof candidate.finishReason === 'string') finish = geminiStopToOpenAI(candidate.finishReason);
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') out += part.text;
      if (isRecord(part.functionCall)) {
        calls.push({
          id: `call_${calls.length}`,
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        });
      }
    }
  }
  return { text: out, toolCalls: calls, finish, inputTokens, outputTokens };
}

/** One JSON answer, in its wire's shape, as an assembled answer: for a
 * streamed request whose provider answered with JSON after all. */
export function jsonToAssembled(wire: 'openai' | 'anthropic' | 'gemini', body: Json): AssembledAnswer {
  if (wire === 'gemini') return jsonToAssembled('openai', geminiToOpenAI(body, 'gemini'));
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    let text = '';
    const toolCalls: ToolCallDraft[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') text += block.text;
      if (block.type === 'tool_use') {
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        });
      }
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
    return {
      text,
      toolCalls,
      finish: stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : 'stop',
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
    };
  }
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ToolCallDraft[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    toolCalls,
    finish: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
  };
}

/** Split text into small chunks for the streamed deltas. */
function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** One assembled answer as an OpenAI SSE stream, ending with usage and [DONE]. */
export function renderOpenAIStream(answer: AssembledAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${model}`,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitPieces(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent(null, chunk({ tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, null)),
    );
    if (call.arguments !== '') {
      frames.push(sseEvent(null, chunk({ tool_calls: [{ index: 0, function: { arguments: call.arguments } }] }, null)));
    }
  }
  frames.push(sseEvent(null, chunk({}, answer.finish)));
  frames.push(
    sseEvent(null, {
      id: `chatcmpl-${model}`,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One assembled answer as an Anthropic SSE stream. */
export function renderAnthropicStream(answer: AssembledAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '') {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitPieces(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  if (index === 0) {
    // An empty answer still needs one block, so the event order stays valid.
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: openAIStopToAnthropic(answer.finish), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
