# entry-free-model-aggregator-01

One Node 22 server that serves the OpenAI Chat Completions API and the
Anthropic Messages API behind a single endpoint, answering every request
with the best-rated free model that meets it, across ten simulated free-tier
providers on three wires (OpenAI, Anthropic, Gemini), with failover when a
provider refuses.

## Run it

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://localhost:8080 node main.ts
```

Print the catalog it registers, without starting a server:

```sh
MAKEFUN_PROMPT_FILE=prompt.txt node main.ts --list
```

No model provider key is needed: the entry makes no model calls of its own
(the competition's harness marks model calls optional), and no key is
required for ranking. All ten providers are the trial's simulated services,
reached through the env.

## How it works

- `lib/brief.ts` reads the brief (provider directory, quality table,
  workload manifest), probes each provider's catalog route in its own wire,
  and builds the one catalog document. The catalog keeps each provider's
  listing order, which the scorer compares in order.
- `main.ts` plans the routing up front: the manifest is walked in order
  (heaviest requests first) and each request takes the best-rated free
  candidate whose provider still has a primary slot (at most three
  primaries per provider), so no provider's unpublished per-trial budget is
  exhausted halfway through. A provider call that answers 402, 429 or 5xx
  fails over to the next candidate in rating order, the refusing provider
  is cooled for the rest of the trial, and when every candidate has refused
  the best candidate is tried once more (a failure-schedule refusal is
  transient). Exactly one provider call ever succeeds per request.
- `lib/translate.ts` translates requests and answers between the OpenAI,
  Anthropic and Gemini wires, keeping every field the client sent intact
  (system prompt, user text and images, tools, stop sequences, max_tokens,
  JSON mode), and re-renders streamed answers in the client's wire with the
  provider's text whole, its usage counts and its stop reason mapped.
  JSON-mode requests are never routed to an Anthropic-wire provider first:
  that wire carries no JSON-mode field.
- `lib/upstream.ts` dials one provider in its own wire.

## Local grade (practice trials, process driver)

```
score 99 on each of trial-0, trial-1 and trial-2
  quality 99, served 100, wire 100, failover 100, catalog 100
  upstream calls 29 / 29 / 31; replay matched on all three;
  L1 regrade reproduced byte for byte
```

Quality 99 rather than 100: a failure-schedule refusal on the top-rated
provider forces one request onto the next-best model (1475 -> 1470), and
the score floors any loss below 100. No request is ever left unserved.

---

Entered on [make.fun](https://make.fun/?ref=p2wx7twh&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
