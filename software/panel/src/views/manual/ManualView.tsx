import { Hand, Undo2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { liveSend } from "../../api/live";
import { paths } from "../../api/paths";
import type { ActionDispatched } from "../../api/types";
import { uuidv7 } from "../../lib/uuidv7";
import { useLiveStore } from "../../state/liveStore";
import { useIsPilot, useSessionStore } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import type { ViewProps } from "../registry";
import { Stick } from "./Stick";
import {
  type Axes,
  axesFromGamepad,
  axesFromSticks,
  isNeutral,
  MANUAL_RATE_HZ,
  NEUTRAL,
  toWire,
} from "./sticks";
import { useGamepadName } from "./useGamepad";

function readGamepad(): Axes | null {
  if (typeof navigator.getGamepads !== "function") return null;
  const pad = Array.from(navigator.getGamepads()).find((p) => p !== null) ?? null;
  return pad ? axesFromGamepad(pad.axes) : null;
}

/**
 * Manual control. "Ta över" pauses the AI's orders and puts the vehicle in LOITER; while this
 * view is open the sticks (on screen or a game controller) are sent 15 times a second over
 * the live socket. Released sticks hold position. If the values stop (view closed, network
 * gone) the ground station stops sending within 1 s and the vehicle stands still. "Lämna
 * tillbaka" gives control back to the AI. A physical RC transmitter always wins.
 */
export default function ManualView({ vehicleId }: ViewProps) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const enabled = useSessionStore((s) => s.config?.features.manual_control !== false);
  const manual = useLiveStore((s) => s.vehicle?.manual ?? null);
  const gamepad = useGamepadName();
  const [takeoverId, setTakeoverId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [axes, setAxes] = useState<Axes>(NEUTRAL);
  const left = useRef({ x: 0, y: 0 });
  const right = useRef({ x: 0, y: 0 });
  const seq = useRef(0);
  const serverActive = manual?.active === true;
  // We steer from the moment our take-over is accepted until the ground reports another one
  // (or none: the effect below then clears ours).
  const steering =
    takeoverId !== null &&
    !(serverActive && manual?.takeover_id != null && manual.takeover_id !== takeoverId);
  const otherActive = serverActive && !steering;

  useEffect(() => {
    if (
      manual &&
      manual.active === false &&
      takeoverId !== null &&
      manual.takeover_id === takeoverId
    ) {
      setTakeoverId(null); // handed back elsewhere, or the ground ended it
    }
  }, [manual, takeoverId]);

  useEffect(() => {
    if (!steering || takeoverId === null) return;
    const id = setInterval(() => {
      const pad = readGamepad();
      const touch = axesFromSticks(left.current, right.current);
      const usePad = pad !== null && !isNeutral(pad) && isNeutral(touch);
      const a = usePad ? pad : touch;
      setAxes(a);
      seq.current += 1;
      liveSend({
        op: "manual",
        takeover_id: takeoverId,
        input_seq: seq.current,
        ...toWire(a),
        source: usePad ? "gamepad" : "touch",
      });
    }, 1000 / MANUAL_RATE_HZ);
    return () => clearInterval(id);
  }, [steering, takeoverId]);

  async function act(action: "takeover" | "handback"): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      const r = await api<ActionDispatched>(paths.vehicleAction(vehicleId, action), {
        method: "POST",
        body: { client_action_id: uuidv7() },
        timeoutMs: 3_000,
      });
      if (action === "takeover") {
        seq.current = 0;
        setTakeoverId(r.action_id);
      } else {
        setTakeoverId(null);
        setAxes(NEUTRAL);
      }
      useUiStore
        .getState()
        .log("info", action === "takeover" ? t("manual.tookOver") : t("manual.handedBack"));
    } catch (e) {
      setError(e instanceof HttpError ? e.message : t("manual.failed"));
    } finally {
      setBusy(false);
    }
  }

  const fmt = (x: number): string => x.toFixed(2);
  return (
    <div className="view-body manual">
      <div className="manual-head">
        {steering || otherActive ? (
          <button
            type="button"
            className="btn btn-primary"
            data-testid="handback"
            disabled={!isPilot || busy}
            onClick={() => void act("handback")}
          >
            <Undo2 aria-hidden="true" size={18} />
            {t("manual.handback")}
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            data-testid="takeover"
            disabled={!isPilot || !enabled || busy}
            onClick={() => void act("takeover")}
          >
            <Hand aria-hidden="true" size={18} />
            {t("manual.takeover")}
          </button>
        )}
        <p className="muted" data-testid="manual-state">
          {!isPilot
            ? t("manual.pilotOnly")
            : steering
              ? t("manual.steering")
              : otherActive
                ? t("manual.otherActive")
                : t("manual.ready")}
        </p>
      </div>
      {error ? <p className="error">{error}</p> : null}
      <p className="muted">{t("manual.explain")}</p>
      <div className="sticks">
        <Stick
          label={t("manual.leftStick")}
          disabled={!steering}
          onChange={(p) => (left.current = p)}
        />
        <Stick
          label={t("manual.rightStick")}
          disabled={!steering}
          onChange={(p) => (right.current = p)}
        />
      </div>
      <p className="muted" data-testid="manual-axes">
        {t("manual.axes", {
          roll: fmt(axes.roll),
          pitch: fmt(axes.pitch),
          yaw: fmt(axes.yaw),
          throttle: fmt(axes.throttle),
        })}
      </p>
      <p className="muted">
        {gamepad ? t("manual.gamepad", { name: gamepad }) : t("manual.noGamepad")}
      </p>
      <p className="muted">{t("manual.rcFirst")}</p>
    </div>
  );
}
