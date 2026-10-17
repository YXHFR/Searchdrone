"""Fixed values of the AI commander. Owner decisions unless a comment says otherwise."""

from __future__ import annotations

from sd_schemas.decide import STOP_CHECK_TARGET_S

__all__ = [
    "BRAIN_HEARTBEAT_S",
    "BRAIN_HEARTBEAT_TIMEOUT_S",
    "BRAIN_TURN_DEADLINE_S",
    "CHAT_TAIL",
    "CLOUD_RETRY_AFTER_S",
    "DEGRADED_MIN_S",
    "EXPOSED_STAGE",
    "FRESH_SNAPSHOT_WAIT_S",
    "HEALTH_INTERVAL_S",
    "HEALTH_TIMEOUT_S",
    "LLM_CALL_TIMEOUT_S",
    "LLM_FAILS_TO_DEGRADE",
    "LLM_SLOW_WARN_S",
    "MAX_ACTIONS_PER_MINUTE",
    "MESSAGE_MAX_CHARS",
    "MIN_ACTION_INTERVAL_S",
    "PROBE_FAILS_TO_DEGRADE",
    "RATIONALE_MAX_CHARS",
    "STOP_CHECK_TARGET_S",
    "SYNC_REJECT_MEMORY_S",
    "TAKEOFF_REQUEST_TTL_S",
    "TOOL_RESULT_MAX_CHARS",
]

BRAIN_HEARTBEAT_S = 1.0
BRAIN_HEARTBEAT_TIMEOUT_S = 5.0  # the ground station's limit (sets brain = down)
BRAIN_TURN_DEADLINE_S = 20.0  # around one whole turn, every backend
LLM_CALL_TIMEOUT_S = 15.0  # one HTTP call to a model, never past the turn deadline
LLM_SLOW_WARN_S = 8.0
LLM_FAILS_TO_DEGRADE = 3
PROBE_FAILS_TO_DEGRADE = 3
DEGRADED_MIN_S = 60.0
HEALTH_INTERVAL_S = 2.0
HEALTH_TIMEOUT_S = 1.0
TAKEOFF_REQUEST_TTL_S = 120.0
CLOUD_RETRY_AFTER_S = 60.0  # after a cloud failure, turns go straight to the local model
EXPOSED_STAGE = 1
# A skill's end, and the pilot's Resume, reach the AI before the snapshot that shows the
# vehicle after them (the onboard computer sends snapshots at 5 Hz; the ground lifts the
# pause only when a snapshot confirms the release). A turn on such an event waits at most
# this long for the newer picture, so its order is not refused for an outdated state.
FRESH_SNAPSHOT_WAIT_S = 2.0
MIN_ACTION_INTERVAL_S = 3.0  # the gate's rate limit: one action per 3 s ...
MAX_ACTIONS_PER_MINUTE = 15  # ... and 15 per minute
CHAT_TAIL = 12
MESSAGE_MAX_CHARS = 600
RATIONALE_MAX_CHARS = 200
TOOL_RESULT_MAX_CHARS = 1500
SYNC_REJECT_MEMORY_S = 600.0
