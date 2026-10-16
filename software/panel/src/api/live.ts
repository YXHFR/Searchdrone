// Wires LiveSocket to the stores. The full resync and the remaining topics come with the
// live-server work; the shell keeps telemetry, vehicle state, system status, chat and
// action outcomes.
import { actionBus } from "../state/actionBus";
import { useChatStore } from "../state/chatStore";
import { useLiveStore } from "../state/liveStore";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { api, HttpError } from "./http";
import { paths } from "./paths";
import type {
  ActionOutcome,
  ChatMessage,
  Mission,
  SystemStatus,
  Telemetry,
  VehicleLive,
} from "./types";
import { type LiveEnvelope, LiveSocket } from "./ws";

export async function loadMission(vehicleId: string): Promise<void> {
  const role = useSessionStore.getState().session?.role;
  try {
    const mission =
      role === "pilot"
        ? await api<Mission>(paths.missionOpen(vehicleId), { method: "POST" })
        : await api<Mission | undefined>(paths.missionActive(vehicleId));
    useLiveStore.getState().patch({ mission: mission ?? null });
  } catch (e) {
    if (!(e instanceof HttpError && e.status === 401)) {
      useUiStore.getState().log("info", "Kunde inte hämta uppdraget");
    }
  }
}

async function resync(vehicleId: string, withMission: boolean): Promise<void> {
  try {
    const v = await api<VehicleLive>(paths.vehicleState(vehicleId));
    useLiveStore
      .getState()
      .patch({ vehicle: v, telemetry: v.telemetry ?? useLiveStore.getState().telemetry });
  } catch {
    // keep what we have; the next event updates it
  }
  if (withMission) await loadMission(vehicleId);
}

function apply(e: LiveEnvelope): void {
  const live = useLiveStore.getState();
  live.patch({ lastWsRxAt: Date.now() });
  switch (`${e.topic}/${e.type}`) {
    case "telemetry/telemetry":
      live.patch({ telemetry: e.data as Telemetry, lastTelemetryAt: Date.now() });
      break;
    case "vehicle/vehicle.state":
      live.patch({ vehicle: e.data as VehicleLive });
      break;
    case "vehicle/vehicle.action":
      actionBus.publish(e.data as ActionOutcome);
      break;
    case "system/system.status":
      live.patch({ system: e.data as SystemStatus });
      break;
    case "mission/mission.updated":
      live.patch({ mission: e.data as Mission });
      break;
    case "chat/chat.message":
      useChatStore.getState().receive(e.data as ChatMessage);
      break;
    case "chat/chat.brain_status":
      live.patch({ brainBusy: (e.data as { state?: string }).state === "busy" });
      break;
    case "alerts/alert": {
      const a = e.data as { severity?: string; message_sv?: string; code?: string };
      useUiStore
        .getState()
        .log(a.severity === "critical" ? "critical" : "info", a.message_sv ?? a.code ?? "");
      break;
    }
    default:
      break; // unknown topics and types are ignored
  }
}

/** Loads the mission once, then keeps live data flowing. Returns a stop function. */
export function startLive(vehicleId: string): () => void {
  void loadMission(vehicleId);
  let epochs = 0;
  const sock = new LiveSocket(
    { vehicleId, missionId: () => useLiveStore.getState().mission?.id ?? null },
    {
      onState: (s) => useLiveStore.getState().setWs(s),
      onEnvelope: apply,
      onNewEpoch: () => {
        epochs += 1;
        void resync(vehicleId, epochs > 1);
      },
    },
  );
  sock.start();
  return () => sock.stop();
}
