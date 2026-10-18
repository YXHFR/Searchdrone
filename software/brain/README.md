# sd-brain (`sd_brain`)

The AI mission commander `sd-brain`. It reads the chat and the situation from the ground
station and answers with text or with one order (a skill from the registry). It can only
act through the ground station's commander API and the safety gate: it has no access to
the flight controller, the database or the operator's buttons, and it can never change
flight-controller parameters, the geofence, the mission limits or its own autonomy mode.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).

## How it decides

- **The cloud model is the AI commander.** With a cloud key (`SD_CLOUD_API_KEY` in `.env`)
  the model in `config/brain.toml` answers (`[cloud]`, any OpenAI-compatible API,
  OpenRouter by default: `z-ai/glm-5.3-flash`, with `anthropic/claude-haiku-5.5` asked
  when the main model returns an error). OpenRouter gets `reasoning: {effort}` (default
  `low`) and `provider: {data_collection: "deny", require_parameters: true}`, both set in
  `[cloud]` and `[cloud.provider]`. Every order goes through the gate.
- **The command mode is the only reserve.** Without network or key, or when the cloud
  does not answer within `call_timeout_s`, there is no AI chat: the fixed command mode
  (pure code, "hjälp" lists the commands) answers the same turn, the heartbeat says
  `engine: "command"` and the panel shows "AI:n svarar inte" with fixed buttons. After a
  failure the cloud rests for `retry_after_s` before it is asked again. The name is
  `command`, not `none`, because something still answers: the command mode and the stop
  words.
- **One action per turn.** The AI keeps its plan in a note and is woken when a step is done.
- **Rules without a model:** "fortsätt" during a stop is answered with the Resume card (the
  chat never releases a stop); a take-off after "Clear around the drone" when the operator
  asked for a start; a fixed command mode ("hjälp") while the model does not answer.
- **Quick stop check.** Every chat line is also asked, in parallel, "does the operator want
  to stop or fly home?" through the decision interface (`sd_brain.decide`: Jev from
  TypeSafe AI or the OpenAI Decisions API, best available first, only with their keys;
  target under 0.5 s). The words `stopp`, `stop` and `flyg hem` always count. A yes is sent to the
  ground station, which runs the same emergency path as the buttons. Without network or a hosted
  key only these fixed words act in the chat (not free phrasings such as "stanna där"). The
  STOP and FLY HOME buttons never depend on the AI.

## Running

```bash
just setup-config        # config/brain.toml, .env, secrets/brain_token
# put SD_CLOUD_API_KEY=... in .env for the AI chat (without it: the command mode)
just ground              # the ground station (in another terminal)
just brain               # the AI commander
```

Other commands: `just brain-eval` (the golden evaluation, no network), `just
brain-eval-cloud` (the same scenarios against the cloud model; needs the key), `just perf`
(the cloud model's answer time per turn, p50/p95; skipped with a line without a key),
`just brain-token` (a new commander token; restart both processes),
`uv run sd-brain config-check` and `uv run sd-brain print-prompt --mission <id>`.

## Settings and keys

`config/brain.toml` (copy of `config/brain.example.toml`); any key can be overridden with
`SD_BRAIN__<SECTION>__<KEY>`. Keys only in `.env`, never in a committed file:

| Name | What |
|---|---|
| `SD_CLOUD_API_KEY` | the cloud model (the AI chat); without it the command mode answers |
| `SD_JEV_API_KEY`, `SD_OPENAI_API_KEY` | optional: the decision engines for the stop check |

The commander token is read from `secrets/brain_token`. sd-brain reads only these names
from `.env`, never the ground station's passwords. Keys are masked in every log line; the
reports of `just perf` and `just brain-eval-cloud` hold latencies and model names only.

Older `config/brain.toml` files from the local-model era still load: `kind = "llm_local"`
becomes `cloud`, and the removed `[llm]` keys and the `local` decision engine are dropped.
