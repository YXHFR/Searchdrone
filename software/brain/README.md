# sd-brain (`sd_brain`)

The AI mission commander `sd-brain`. It reads the chat and the situation from the ground
station and answers with text or with one order (a skill from the registry). It can only
act through the ground station's commander API and the safety gate: it has no access to
the flight controller, the database or the operator's buttons, and it can never change
flight-controller parameters, the geofence, the mission limits or its own autonomy mode.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).

## How it decides

- **Best available engine.** With a cloud key (`SD_CLOUD_API_KEY` in `.env`) and a model
  in `config/brain.toml` (`[cloud] model`, any OpenAI-compatible API, OpenRouter by
  default), the cloud model answers first. When it fails or does not answer in time, the
  local model (Qwen3-VL-4B in llama-server) answers in the same turn. Without a key
  everything runs locally. Both get the same tools, and every order goes through the gate.
- **One action per turn.** The AI keeps its plan in a note and is woken when a step is done.
- **Rules without a model:** "fortsätt" during a stop is answered with the Resume card (the
  chat never releases a stop); a take-off after "Clear around the drone" when the operator
  asked for a start; a fixed command mode ("hjälp") while the model does not answer.
- **Quick stop check.** Every chat line is also asked, in parallel, "does the operator want
  to stop or fly home?" through the decision interface (`sd_brain.decide`: the local model,
  Jev from TypeSafe AI or the OpenAI Decisions API, best available first; target under
  0.5 s). The words `stopp`, `stop` and `flyg hem` always count. A yes is sent to the
  ground station, which runs the same emergency path as the buttons. The STOP and FLY HOME
  buttons never depend on the AI.

## Running

```bash
just setup-config        # config/brain.toml, .env with SD_LLM_API_KEY, secrets/brain_token
just models-fetch        # the local model (about 2.5 GB) into models/, sha256 checked
just llm-up              # llama-server on 127.0.0.1:8091 (needs an NVIDIA GPU and Docker)
just ground              # the ground station (in another terminal)
just brain               # the AI commander
```

Other commands: `just brain-eval` (the golden evaluation, no GPU), `just brain-eval-model`
(the model evaluation on your computer), `just check-model` (5 minutes of calls and the
graphics memory), `just brain-token` (a new commander token; restart both processes),
`uv run sd-brain config-check` and `uv run sd-brain print-prompt --mission <id>`.

## Settings and keys

`config/brain.toml` (copy of `config/brain.example.toml`); any key can be overridden with
`SD_BRAIN__<SECTION>__<KEY>`. Keys only in `.env`, never in a committed file:

| Name | What |
|---|---|
| `SD_LLM_API_KEY` | key between sd-brain and llama-server (made by `just setup-config`) |
| `SD_CLOUD_API_KEY` | optional: the cloud model |
| `SD_JEV_API_KEY`, `SD_OPENAI_API_KEY` | optional: the decision engines for the stop check |

The commander token is read from `secrets/brain_token`. sd-brain reads only these names
from `.env`, never the ground station's passwords.
