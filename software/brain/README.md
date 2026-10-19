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
  `low`) and `provider: {data_collection: "deny", require_parameters: true}`, plus
  `zdr: true` outside the profile `sim`, all set in `[cloud]` and `[cloud.provider]`.
  Every order goes through the gate.
- **What leaves the machine.** With `SD_CLOUD_API_KEY` set, each turn (operator chat and
  vehicle state) goes to OpenRouter and on to the main model's provider, and on an error
  also to the fallback model's provider. `data_collection = "deny"` keeps it out of
  training and storage. With `profile = "bench"` or `"field"` (real flights, real search
  areas) `zdr` defaults to `true`: only zero-data-retention endpoints, for the main and
  the fallback model alike (both have such endpoints on OpenRouter). In `sim` it defaults
  to `false`; an explicit `zdr` in `[cloud.provider]` wins either way (`config-check`
  and the start warn about `zdr = false` on bench or field). If no
  zero-retention endpoint answers, the command mode answers the turn as for any other
  cloud failure. `fallback_model = ""` keeps it to one vendor.
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
brain-eval-cloud` (the same scenarios against the cloud model; needs the key), `just
brain-eval-sim` and `just brain-scores` (simulator exams, below), `just perf`
(the cloud model's answer time per turn, p50/p95; skipped with a line without a key),
`just brain-token` (a new commander token; restart both processes),
`uv run sd-brain config-check` and `uv run sd-brain print-prompt --mission <id>`.

## Simulator exams

`sd-brain eval --backend sim --flights 3` (`just brain-eval-sim flights=3`) flies random
search scenarios with an answer key (a missing person somewhere in the charter, false
sightings, sometimes a STOP from the pilot) in a fast 2D world (`sd_brain.simeval`). The
real commander loop decides through the cloud model; the world flies the orders the gate
allows as the executor would. The AI never steers the drone itself.

- **Grade per flight** (`data/brain-scores/<day>/<flight>.json`): safety (rejected or
  forbidden orders, orders during STOP, geofence breaches), found or not and the time to
  find, coverage of the area, pilot questions and the cost in kronor. A safety failure
  always fails the flight (`passed: false`, `total: 0`).
- **Cost cap: 30 kr per day**, counted on the price OpenRouter states for every call
  (`usage.cost`, converted at a deliberately high dollar rate) in
  `data/brain-eval/cost-ledger.json`, shared by all runs on the machine. `--max-cost-sek`
  caps one run lower. At the cap, without a key, when the provider refuses the key, or when
  the cloud fails, a fixed search script flies the rest and the flight is not an exam;
  `fallback_reason` says why (`budget`, `key_invalid`, `cloud_error`). A flight where the
  provider did not price every call shows its cost as not measured (`cost_measured:
  false`, "ej uppmätt"); no price is guessed.
- **Example bank:** exams that pass with at least 75 points (`good-<id>`) and exams that
  fail (`fail-<id>`) are saved (the model's answers, and per turn the prompt, tool calls
  and outcome) in `data/example-bank/` (gitignored; a folder git does not ignore is
  refused) or in an R2 bucket in the EU when all four `SD_BANK_R2_ENDPOINT`
  (`https://<account>.eu.r2.cloudflarestorage.com`), `SD_BANK_R2_BUCKET`,
  `SD_BANK_R2_ACCESS_KEY_ID` and `SD_BANK_R2_SECRET_ACCESS_KEY` are set.
- **Replay fixtures:** `sd-brain bank export-golden data/example-bank/good-<id>.json.gz
  --name SIM-03` writes `tests/golden/sim/SIM-03.json`: only the scenario seed and the
  model's answers (no prompt, no ids, no times; a flight whose answers look like they hold
  a key or personal data is refused). `just brain-eval` flies every fixture again through
  the real loop and the gate without a network and fails when the flight goes another way.
- **Failure list:** `sd-brain bank failures` lists the failed exams with what went wrong
  (safety, not found, false marks, coverage, questions) and the seed to fly them again;
  that is where the commander's instructions need work.
- **Trend:** `just brain-scores` prints the mean grade, pass rate, found rate, safety
  failures and cost per flight per day and model, and flags a day that got worse.
- **For a simulator farm:** `make_decision_maker` gives the decider for one flight (a
  `CommanderBackend`; check `exam` afterwards) and `score_flight` grades a `FlightLog`.
  `sd-brain eval grade --round <dir>` grades a farm round: it reads `round.json`
  (`simfarm.round/v1`), `truth.json` and per run `flight.json` (`simeval.farm.FarmFlight`)
  and `brain.json` (`brain_record`), and writes `grade.json` with `commander_used`,
  `is_exam`, `safety_ok`, `found`, `cost_sek` (null when not measured) and
  `fallback_reason`.
- **Nightly:** the `brain-nightly` workflow runs three flights (at most 5 kr) from main when
  the repository has the secret `OPENROUTER_API_KEY`, and uploads the grades as an artifact.

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
