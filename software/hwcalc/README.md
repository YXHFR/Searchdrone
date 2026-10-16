# sd-hwcalc (`sd_hwcalc`)

Hardware calculations (mass, flight time) checked against `hardware/drone.yaml`.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just hw-check` (all checks: `just check`).

`python -m sd_hwcalc check <drone.yaml> --limits <limits.yaml>` prints a JSON report
(total mass, hover power, usable energy, hover time, hover thrust fraction,
thrust-to-weight, pass, failures) and a short summary on stderr. Exit code 1 when a limit
is broken or a file does not validate.
