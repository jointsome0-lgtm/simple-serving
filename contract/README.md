# Shared cases

`cases-v1.json` holds the cases of contract v1 ([docs/contract-v1.md](../docs/contract-v1.md)). The gateway's Python
tests and the bot's TypeScript tests read the same file.

## Who runs what

- The gateway's tests start the gateway with the `service` block as its configuration, in a real ASGI server, in front
  of the fake engine. For each step they script the fake engine with `engine`, send `request`, and compare the answer
  with `response` and what the engine received with `engine_receives`.
- A client's tests start a fake gateway that answers each step with `response`. They call the client and compare what
  it returns with `result`. A client skips the steps it has no call for; the bot today has none for `base: control`.
- A static case cannot describe timing. `scenarios` lists what each implementation tests on its own (contract
  section 14).

## The service block

- `alias`, `context_tokens`, `body_limit_bytes`.
- `keys`: each test key with its label, the classes it may use, its default class (`null` for a key without classes),
  `scopes` (may name cache scopes) and `control` (may call the control routes).
- `limits`: for each class `active`, `waiting`, `input_tokens` (`null` means the context), `max_tokens` and `wall_s`;
  `external_per_key`; `engine.active`, the places in the engine for all classes together.
- `count_limits`, `engine_priority` (lower goes first, as in vLLM), `drain_deadline_s`.

The values are small so that tests reach them. They are not the provisional values of section 7.

## A case

A case has a `name`, an `about` and `steps`. Every case starts on a fresh gateway: a new boot, status `ready`, drain
generation 0. Its steps run in order on that gateway, so a drain or a captured value carries over to the next step.
A step may have a `name`, which other steps of the case refer to.

### request

- `base`: `public` or `control`, the listener.
- `method`, `path`.
- `key`: the bearer key. `null` sends no `Authorization` header.
- `headers`: more headers.
- The body is one of:
  - `body_patch`: `defaults.chat_body` with these top-level fields replaced. `null` is sent as `null`.
  - `body`: sent as it is.
  - `raw_body`: these characters, which are not JSON.
  - `raw_body_bytes`: that many bytes of `a`. With `chunked: true` they go with chunked transfer encoding and no
    `Content-Length`.
  - none: no body.

### engine

What the fake engine does in this step. `null` means the gateway must not call the engine at all.

- `input_tokens`: the engine's count of the prompt, in its tokenize answer and as `prompt_tokens` of its generation.
- `events`: the stream of one generation, in order. `role`, `content`, `reasoning` (the engine's own name for
  reasoning) and `finish_reason` fill one chunk; `{}` is an empty delta; `{"break": true}` drops the connection
  without an end.
- `usage`: `completion_tokens` and, when given, `cached_tokens` of the engine's usage chunk.
- `generate_status`: the engine answers the generation with this HTTP status and an error body, and streams nothing.

### engine_receives

What the engine received in this step. Only the gateway's tests check it.

- `calls`: the engine calls in order, `tokenize` and `generate`.
- `priority`: the engine priority of the generation.
- `cache_salt`: `"present"`, `{"same_as": "<step>"}` or `{"differs_from": ["<step>", ...]}`, compared with the salt of
  the named steps of the same case.
- `response_format`: `"json_schema"` means the generation carried the client's schema unchanged.
- `headers_absent`: headers that must not reach the engine.

### response

What the gateway answers.

- `status`.
- `error`: the body is exactly `{"error": {"code": "<error>"}}`.
- `json`: the body. Its keys must match exactly, unless `json_subset` is `true`; then only the keys shown are compared.
- `chunks`: the `data:` events of the stream before `[DONE]`, in order. `choices` and `usage` are compared exactly,
  and a `finish_reason` of `null` counts as absent. Every chunk must also have `model` equal to the alias, which the
  cases leave out. Other top-level keys, such as `id`, `object` and `created`, are not compared.
- `done`: whether the stream ends with `data: [DONE]`.
- `error_event`: the code of the error event that ends a stream without `[DONE]`.
- `capture`: `{"<name>": "<key>"}` keeps that key of the answer for later steps, where `"$<name>"` stands for it.

Placeholders in expected values:

- `"present"`: any value except `null`.
- `"measurements"`: an object with exactly `wait_ms`, `first_token_ms` and `total_ms`, non-negative integers, with
  `wait_ms` ≤ `first_token_ms` ≤ `total_ms`.
- `"$<name>"`: the captured value.

### result

What a client makes of the response, in neutral names that each client maps to its own types:

- `text`, `reasoning`, `finish_reason`, and `usage` with `input`, `output` and `cached` (`null` means unknown);
- `input_tokens` of a count;
- `model` and `context_tokens` from `/v1/models`, `status` from `/v1/state`;
- `error`: the gateway's code. Each client maps the codes to its own errors; the bot's table is in its adapter.

## Changing the cases

A change that breaks a client is a new contract version and a new file. simple-story-chat keeps a pinned copy with
the contract version, the commit of this repository and the SHA-256 of the file, and updates it on purpose.
