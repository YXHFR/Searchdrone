"""``just sim-farm``: run farm rounds until stopped (Ctrl+C), politely, from the newest main.

Settings (``key=value``, over ``config/simfarm.toml``): ``commander=script|sd-brain``,
``rounds=N`` (0 = until stopped), ``max_minutes=N`` (0 = no time cap; at the cap the
running round is stopped and removed, and the farm exits 0), ``seed=N`` (the first
round's seed; then N+1, ...),
``idle_only=true``, ``update=off`` (run rounds from this checkout instead of the newest
main; for trying out a change, and refused with an R2 key, which the round would then
see) and the other ``[farm]`` keys.

Every round:

1. upload what is waiting (with an R2 key), keep the disk budget, and wait for a turn
   (``host.Gate``: busy GPU or CPU, the duty cycle, ``idle_only``);
2. bring the worktree ``build/simfarm/main`` to ``origin/main`` and sync it (offline:
   the last commit it had);
3. run the round there in a process group of its own under ``nice``
   (``sd_sim.farm.round``), into ``build/simfarm/rounds/.partial-<id>/``;
4. a passed round becomes ``rounds/<id>/`` and is uploaded; a failed one goes to
   ``failed/<id>/`` with its ``round.json`` only and counts towards the stop.

Ctrl+C (or SIGTERM) stops the running round, removes its half-made folder and uploads
nothing from it. ``max_failures`` failed rounds in a row stop the farm with their reasons.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from sd_sim import _args, paths, procs
from sd_sim.farm import config as farm_config
from sd_sim.farm import people, status
from sd_sim.farm import round as farm_round
from sd_sim.farm.host import CHECK_EVERY_S, Gate
from sd_sim.farm.store import Ledger, R2Store, UploadError

ALLOWED_KEYS = (
    "commander",
    "rounds",
    "max_minutes",
    "seed",
    "update",
    "idle_only",
    "busy_gpu_pct",
    "busy_cpu_load",
    "max_duty",
    "max_failures",
)
CONFIG_KEYS = (
    "commander",
    "idle_only",
    "busy_gpu_pct",
    "busy_cpu_load",
    "max_duty",
    "max_failures",
)
GIT_TIMEOUT_S = 300
SYNC_TIMEOUT_S = 1800
ROUND_TIMEOUT_S = 3 * 3600
STOP_GRACE_S = 30
# Local, gitignored files the round needs from the owner's checkout (keys, local site).
LINKED = (".env", "secrets")
# The uv settings a round's ``uv run`` may use; never uv's index or publish credentials.
UV_ENV_KEYS = (
    "UV_CACHE_DIR", "UV_PYTHON", "UV_PYTHON_INSTALL_DIR", "UV_OFFLINE", "UV_NO_SYNC",
    "UV_NATIVE_TLS", "UV_SYSTEM_CERTS", "UV_LINK_MODE", "UV_CONCURRENT_DOWNLOADS",
)  # fmt: skip
# What the round process may see of the farm's environment (as ``procs.clean_env``): uv,
# the network proxy and certificates, and the display for a Gazebo window. Keys are read
# from the linked ``.env`` by whoever needs them, never passed on here.
NET_ENV_KEYS = (
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)  # fmt: skip
# Where the simulator is installed (``tools/setup/wsl-setup.sh sim``).
SIM_ENV_KEYS = (
    "GZ_VERSION", "GZ_IP", "GZ_SIM_SYSTEM_PLUGIN_PATH",
    "ARDUPILOT_HOME", "SD_ARDUCOPTER_BIN", "SD_ARDUPILOT_GAZEBO_HOME",
)  # fmt: skip
ROUND_ENV_KEYS = (*procs.GRAPHICS_ENV_KEYS, *NET_ENV_KEYS, *SIM_ENV_KEYS, *UV_ENV_KEYS)
# The local config a round reads: the simulator's and sd-brain's (its cloud model). Never
# the ground station's or the onboard side's, so no vehicle address reaches a round: it
# flies SITL on loopback only.
ROUND_CONFIGS = ("config/sim.toml", "config/brain.toml")
# What git may see of the environment: its login and the network, never a variable that
# points it at another repository, index or config.
GIT_ENV_KEYS = (
    *NET_ENV_KEYS,
    "SSH_AUTH_SOCK", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_ASKPASS", "SSH_ASKPASS",
    "GIT_TERMINAL_PROMPT", "XDG_CONFIG_HOME",
)  # fmt: skip


def say(msg: str) -> None:
    print(f"[simfarm {time.strftime('%H:%M:%S')}] {msg}", flush=True)  # noqa: T201 - command-line output


class Stop:
    """Set by Ctrl+C or SIGTERM; the loop checks it between steps."""

    def __init__(self) -> None:
        self.requested = False
        self.proc: subprocess.Popen[bytes] | None = None

    def handler(self, _sig: int, _frame: FrameType | None) -> None:
        if not self.requested:
            say("stoppar: avbryter varvet och städar (inget laddas upp från det)")
        self.requested = True
        if self.proc is not None and self.proc.poll() is None:
            _signal_group(self.proc, signal.SIGINT)


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def _git(repo: Path, *args: str, timeout: float = GIT_TIMEOUT_S) -> str:
    env = procs.clean_env(passthrough=GIT_ENV_KEYS)
    out = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "-C", str(repo), *args],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return out.stdout.strip()


@dataclass
class Worktree:
    """``build/simfarm/main``: a detached worktree at ``origin/main``, never the owner's
    own checkout, so the farm never touches work in progress."""

    repo: Path
    path: Path
    run: Callable[..., Any] = subprocess.run
    notes: list[str] = field(default_factory=list)

    def update(self) -> str:
        try:
            _git(self.repo, "fetch", "--quiet", "origin", "main")
        except (OSError, subprocess.SubprocessError) as exc:
            self.notes.append(f"kunde inte hämta main ({type(exc).__name__}); kör senast hämtade")
        if not (self.path / ".git").exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _git(self.repo, "worktree", "prune")
            _git(self.repo, "worktree", "add", "--detach", "--force", str(self.path), "origin/main")
        # checkout --force and clean -x below must never reach the owner's own checkout
        # (git looks for a repository in the parent folders when this one is broken).
        top = Path(_git(self.path, "rev-parse", "--show-toplevel")).resolve()
        if top != self.path.resolve():
            raise OSError(f"{self.path} är inte farmens egen worktree (git hittar {top})")
        _git(self.path, "checkout", "--quiet", "--detach", "--force", "origin/main")
        _git(
            self.path,
            "clean",
            "-ffdxq",
            "-e",
            "/build/",
            "-e",
            "/.venv/",
            "-e",
            "/config/",
            "-e",
            "/runs/",
        )
        self.link_local()
        self.run(
            ["uv", "sync", "--all-packages", "--frozen", "--quiet"],
            cwd=self.path,
            check=True,
            timeout=SYNC_TIMEOUT_S,
        )
        return _git(self.path, "rev-parse", "HEAD")

    def link_local(self) -> None:
        """Point the worktree's local config (``ROUND_CONFIGS``) and secrets at the owner's
        checkout. Not the farm's own config: its R2 key is for the loop's uploads, never a
        round's; a config a round must not read is removed if an earlier farm linked it."""
        for stale in (self.path / "config").glob("*.toml"):
            rel = f"config/{stale.name}"
            if stale.is_symlink() and rel not in ROUND_CONFIGS:
                stale.unlink()
        for name in (*ROUND_CONFIGS, *LINKED):
            src, dst = self.repo / name, self.path / name
            if not src.exists():
                continue
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)


@dataclass
class Farm:
    repo: Path
    cfg: farm_config.FarmConfig
    gate: Gate
    stop: Stop
    rounds_wanted: int = 0
    max_minutes: float = 0.0
    seed: int = 0
    update: bool = True
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    round_timeout_s: float = ROUND_TIMEOUT_S
    streak: int = 0
    errors: list[str] = field(default_factory=list)
    commit: str = ""
    deadline: float | None = None
    interrupted: bool = False

    def time_up(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline

    def halting(self) -> bool:
        """Ctrl+C, SIGTERM or the time cap."""
        return self.stop.requested or self.time_up()

    @property
    def base(self) -> Path:
        return status.farm_dir(self.repo)

    @property
    def rounds_dir(self) -> Path:
        return self.base / "rounds"

    def state(self, state: str, reason: str | None = None) -> None:
        status.write_state(
            self.base / "state.json",
            {
                "pid": os.getpid(),
                "state": state,
                "reason": reason,
                "commit": self.commit,
                "streak": self.streak,
                "commander": self.cfg.commander,
                "upload": "r2" if self.cfg.r2 is not None else "local",
            },
        )

    def store(self) -> R2Store | None:
        if self.cfg.r2 is None:
            return None
        return R2Store(self.cfg.r2, Ledger(self.base / "uploaded.jsonl"))

    def upload_backlog(self) -> None:
        store = self.store()
        if store is None:
            return
        for d in sorted(p for p in self.rounds_dir.glob("[!.]*") if p.is_dir()):
            if self.halting():
                return
            try:
                if store.upload_round(d):
                    say(f"uppladdat till R2: {d.name}")
            except UploadError as exc:
                say(f"uppladdningen väntar: {exc}")
                return

    def make_room(self) -> str | None:
        """Keep the disk budget: remove uploaded rounds, oldest first; the reason to
        pause when that is not enough."""
        problem = self.gate.disk_ok()
        if problem is None:
            return None
        uploaded = Ledger(self.base / "uploaded.jsonl").rounds if self.cfg.r2 else set()
        for d in sorted(p for p in self.rounds_dir.glob("[!.]*") if p.is_dir()):
            if d.name in uploaded:
                shutil.rmtree(d, ignore_errors=True)
                say(f"tog bort uppladdat varv lokalt: {d.name}")
                if self.gate.disk_ok() is None:
                    return None
        return problem

    def worktree(self) -> Path:
        if not self.update:
            self.commit = _git(self.repo, "rev-parse", "HEAD")
            return self.repo
        wt = Worktree(self.repo, self.base / "main")
        self.commit = wt.update()
        for note in wt.notes:
            say(note)
        return wt.path

    def round_argv(self, spec_path: Path) -> list[str]:
        return [
            "nice", "-n", str(self.cfg.limits.nice),
            "uv", "run", "--no-sync", "--all-packages",
            "python", "-m", "sd_sim.farm", "round", f"spec={spec_path}",
        ]  # fmt: skip

    def run_round(self, where: Path, seed: int) -> tuple[int, Path, farm_round.RoundSpec]:
        spec = farm_round.RoundSpec.new(
            seed, self.cfg.commander, self.commit, worlds=self.cfg.worlds
        )
        partial = self.rounds_dir / f".partial-{spec.round_id}"
        partial.mkdir(parents=True)
        spec.write(partial / "spec.json")
        argv = self.round_argv(partial / "spec.json")
        env = round_env(where)
        proc = subprocess.Popen(argv, cwd=where, env=env, start_new_session=True)  # noqa: S603
        self.stop.proc = proc
        try:
            code = self._wait(proc)
        finally:
            self.stop.proc = None
        if code is None:
            code = farm_round.EXIT_ERROR
            farm_round.write_result(
                partial, spec, {"status": "error", "error": "varvet tog för lång tid"}
            )
        return code, partial, spec

    def _wait(self, proc: subprocess.Popen[bytes]) -> int | None:
        """The round's exit code; ``None`` when it ran out of time. A round that does not
        stop within ``STOP_GRACE_S`` of a stop or a timeout is killed with its group."""
        start = self.clock()
        stopping: float | None = None
        late = False
        while True:
            try:
                code = proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = None
            if code is not None:
                return None if late else code
            now = self.clock()
            if stopping is None:
                late = now - start > self.round_timeout_s
                self.interrupted = not late and self.halting()
                if late or self.interrupted:
                    _signal_group(proc, signal.SIGINT)
                    stopping = now
            elif now - stopping > STOP_GRACE_S:
                _signal_group(proc, signal.SIGKILL)
                proc.wait()
                return None if late else proc.returncode

    def settle(self, code: int, partial: Path, spec: farm_round.RoundSpec, seconds: float) -> None:
        """Keep a passed round as data, a failed one as an error."""
        result = farm_round.read_result(partial) or {
            "status": "error",
            "error": f"varvet avslutades med kod {code} utan round.json",
        }
        qc = result.get("qc") or {}
        row: dict[str, Any] = {
            "round_id": spec.round_id,
            "seed": spec.seed,
            "commit": spec.commit,
            "status": result.get("status"),
            "error": result.get("error"),
            "images": result.get("images", 0),
            "labels": qc.get("labels", 0),
            "seconds": round(seconds, 1),
            "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
            "grade": result.get("grade"),
        }
        if result.get("status") == "ok" and code != farm_round.EXIT_OK:
            # round.json says ok, the process did not: it broke after writing it.
            row["status"] = "error"
            row["error"] = f"varvet avslutades med kod {code} efter ett godkänt round.json"
        if code == farm_round.EXIT_OK and row["status"] == "ok":
            final = self.rounds_dir / spec.round_id
            partial.rename(final)
            shutil.rmtree(final / "work", ignore_errors=True)
            self.streak = 0
            say(
                f"varv klart: {row['images']} bilder, {row['labels']} personrutor ({spec.round_id})"
            )
        else:
            failed = self.base / "failed" / spec.round_id
            failed.mkdir(parents=True, exist_ok=True)
            for name in ("round.json", "spec.json"):
                if (partial / name).is_file():
                    shutil.copyfile(partial / name, failed / name)
            shutil.rmtree(partial, ignore_errors=True)
            self.streak += 1
            self.errors.append(f"{spec.round_id}: {row['error']}")
            say(f"varvet blev fel ({self.streak} i rad): {row['error']}")
        status.append_round(self.base / "rounds.jsonl", row)

    def run(self) -> int:
        for warning in self.cfg.warnings:
            say(f"varning: {warning}")
        say(
            f"startar: beslutsfattare {self.cfg.commander}, "
            + ("R2 i EU" if self.cfg.r2 else "bara lokalt (build/simfarm/)")
        )
        if self.max_minutes > 0:
            self.deadline = self.clock() + self.max_minutes * 60.0
        # A round cut short by a crash or a kill of the farm itself is never resumed.
        for stale in self.rounds_dir.glob(".partial-*"):
            shutil.rmtree(stale, ignore_errors=True)
        done = 0
        seed = self.seed
        while not self.halting() and (self.rounds_wanted == 0 or done < self.rounds_wanted):
            self.upload_backlog()
            reason = self.make_room() or self.gate.reason_to_wait()
            if reason is not None:
                self.state("paused", reason)
                say(f"paus: {reason}")
                self._nap(CHECK_EVERY_S)
                continue
            self.state("running", "uppdaterar main")
            try:
                where = self.worktree()
            except (OSError, subprocess.SubprocessError) as exc:
                self.streak += 1
                self.errors.append(f"uppdatering: {exc}")
                say(f"kunde inte uppdatera main: {exc}")
                if self.streak >= self.cfg.max_failures:
                    break
                self._nap(CHECK_EVERY_S)
                continue
            self.state("running", f"varv med frö {seed}")
            started = self.clock()
            code, partial, spec = self.run_round(where, seed)
            took = self.clock() - started
            self.gate.round_took(took)
            if self.stop.requested or self.interrupted:
                shutil.rmtree(partial, ignore_errors=True)
                break
            if code == farm_round.EXIT_NOT_BUILT:
                result = farm_round.read_result(partial) or {}
                shutil.rmtree(partial, ignore_errors=True)
                self.state("stopped", "inte byggt än")
                say(f"inte byggt än: {result.get('error', 'varvets körare saknas')}")
                return _args.NOT_BUILT_EXIT
            self.settle(code, partial, spec, took)
            done += 1
            seed += 1
            if self.streak >= self.cfg.max_failures:
                break
            self.upload_backlog()
        if self.streak >= self.cfg.max_failures:
            self.state("stopped", f"{self.streak} fel i rad")
            say(f"stoppar efter {self.streak} fel i rad:")
            for err in self.errors[-self.streak :]:
                say(f"  {err}")
            return 1
        if self.stop.requested:
            self.state("stopped", "avbruten")
            say("stoppad")
        elif self.time_up():
            self.state("stopped", "tidstaket nått")
            say(f"tidstaket på {self.max_minutes:g} min nått efter {done} varv")
        else:
            self.state("stopped", "klar")
            say(f"klar efter {done} varv")
        return 0

    def _nap(self, seconds: float) -> None:
        end = self.clock() + seconds
        if self.deadline is not None:
            end = min(end, self.deadline)
        while not self.stop.requested and self.clock() < end:
            self.sleep(min(1.0, end - self.clock()))


# sd-brain's cost ledger in the owner's checkout: the day's cap for the cloud model.
COST_LEDGER = Path("data") / "brain-eval" / "cost-ledger.json"


def round_env(where: Path) -> dict[str, str]:
    """The round's environment: an allowlist, never the farm's whole environment."""
    extra = {
        "GZ_SIM_RESOURCE_PATH": _resource_path(where),
        "SD_FARM_COST_LEDGER": str(paths.REPO / COST_LEDGER),
    }
    return procs.clean_env(extra, ROUND_ENV_KEYS)


def _resource_path(where: Path) -> str:
    """Gazebo finds the person models under the owner's checkout's assets."""
    parts = [str(people.assets_dir(paths.REPO)), str(people.assets_dir(where))]
    old = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    return ":".join([*dict.fromkeys(parts), *([old] if old else [])])


def _int(values: dict[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, default))
    except ValueError:
        _args.fail(f"sd_sim.farm: {key}= ska vara ett heltal")


def _float(values: dict[str, str], key: str, default: float) -> float:
    try:
        out = float(values.get(key, default))
    except ValueError:
        _args.fail(f"sd_sim.farm: {key}= ska vara ett tal")
    if not math.isfinite(out) or out < 0:
        _args.fail(f"sd_sim.farm: {key}= ska vara ett tal, 0 eller större")
    return out


def main(argv: Sequence[str], repo: Path = paths.REPO) -> int:
    values = _args.parse(argv, ALLOWED_KEYS, "sd_sim.farm")
    try:
        cfg = farm_config.load({k: v for k, v in values.items() if k in CONFIG_KEYS}, repo)
    except farm_config.FarmConfigError as exc:
        _args.fail(f"sd_sim.farm: {exc}")
    update = values.get("update", "on").lower() not in ("off", "0", "false", "no")
    if not update and cfg.r2 is not None:
        # Rounds would run in this checkout, where config/simfarm.toml holds the R2 key.
        _args.fail("sd_sim.farm: update=off går inte med en R2-nyckel i config/simfarm.toml")
    missing = people.missing(repo)
    if missing:
        say(f"personmodellerna saknas ({', '.join(missing)}); hämtar dem en gång")
        try:
            people.fetch(repo, say=say)
        except (OSError, people.PeopleError) as exc:
            say(f"kunde inte hämta personmodellerna: {exc}. Kör just sim-farm-setup med nät.")
            return 1
    stop = Stop()
    signal.signal(signal.SIGINT, stop.handler)
    signal.signal(signal.SIGTERM, stop.handler)
    farm = Farm(
        repo=repo,
        cfg=cfg,
        gate=Gate(cfg.limits, status.farm_dir(repo) / "rounds"),
        stop=stop,
        rounds_wanted=_int(values, "rounds", 0),
        max_minutes=_float(values, "max_minutes", 0.0),
        seed=_int(values, "seed", random.SystemRandom().randrange(1, 1_000_000_000)),
        update=update,
    )
    return farm.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
