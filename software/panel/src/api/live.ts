// Wires LiveSocket to the stores: the resync over REST on every new server epoch, then the
// live events. Events that arrive while a resync runs are held back and applied after it
// in seq order, so a snapshot can never overwrite something newer.
import i18next from "i18next";
import { noteLive } from "../lib/testHooks";
import { alertView, findHoldEndedView, isSticky, LINK_DOWN_CODES } from "../safety/gnss";
import { actionBus } from "../state/actionBus";
import { lastChatSeq, useChatStore } from "../state/chatStore";
import { useLiveStore } from "../state/liveStore";
import { type ClearanceView, type SectorFeature, useMissionStore } from "../state/missionStore";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { api, HttpError } from "./http";
import { paths } from "./paths";
import type {
  ActionOutcome,
  Annotation,
  Approval,
  Charter,
  ChatMessage,
  Command,
  Find,
  Mission,
  MissionDetail,
  OperatorLatch,
  SystemStatus,
  Telemetry,
  VehicleLive,
} from "./types";
import { type LiveEnvelope, LiveSocket } from "./ws";

/** Sector progress has no live event: it is read again this often while the mission flies. */
export const SECTOR_POLL_MS = 3_000;
const FLYING = new Set<Mission["status"]>(["ACTIVE", "PAUSED", "HELD", "FAILSAFE", "RETURNING"]);

interface ChatPage {
  items: ChatMessage[];
  next_before: number | null;
}

interface TrackFeature {
  geometry?: { coordinates?: number[][] };
}

let socket: LiveSocket | null = null;

/** Sends one message on the live socket (stick values). False when it is not open. */
export function liveSend(msg: { op: string; [k: string]: unknown }): boolean {
  return socket?.sendOp(msg) ?? false;
}

const quietly = async <T>(p: Promise<T>): Promise<T | null> => {
  try {
    return await p;
  } catch (e) {
    if (!(e instanceof HttpError && e.status === 401)) {
      useUiStore.getState().log("info", "Kunde inte hämta data från markstationen");
    }
    return null;
  }
};

/** The mission per role: a pilot opens (or gets) the vehicle's mission, an observer reads it. */
export async function loadMission(vehicleId: string): Promise<Mission | null> {
  const role = useSessionStore.getState().session?.role;
  const mission = await quietly(
    role === "pilot"
      ? api<Mission>(paths.missionOpen(vehicleId), { method: "POST" })
      : api<Mission | undefined>(paths.missionActive(vehicleId)),
  );
  setMission(mission ?? null);
  return mission ?? null;
}

function setMission(mission: Mission | null): void {
  const live = useLiveStore.getState();
  const changed = (live.mission?.id ?? null) !== (mission?.id ?? null);
  live.patch({ mission });
  if (changed) {
    useMissionStore.getState().reset(mission?.id ?? null);
    useChatStore.getState().clear();
    socket?.resubscribe();
  }
}

/** Everything that belongs to the mission: detail, charters, pending approvals, sectors,
 * annotations, the finds, the latest chat page and the track, in that order. */
async function loadMissionData(missionId: string): Promise<void> {
  const m = useMissionStore.getState();
  const detail = await quietly(api<MissionDetail>(paths.mission(missionId)));
  if (detail) {
    useLiveStore.getState().patch({ mission: detail.mission, latch: detail.latch });
    if (detail.charter) m.upsertCharter(detail.charter);
    m.setClearance(
      detail.clearance
        ? { status: detail.clearance.status, valid_until: detail.clearance.valid_until }
        : { status: "none", valid_until: null },
    );
    if (detail.active_command) m.upsertCommand(detail.active_command);
    m.setActiveCommand(detail.active_command?.id ?? null);
  }
  const charters = await quietly(api<Charter[]>(paths.missionCharters(missionId)));
  if (charters) m.setCharters(charters);
  const approvals = await quietly(api<Approval[]>(paths.missionApprovals(missionId, "pending")));
  if (approvals) m.setPendingApprovals(approvals);
  await loadSectors(missionId);
  const anns = await quietly(api<Annotation[]>(paths.missionAnnotations(missionId)));
  if (anns) m.setAnnotations(anns);
  const finds = await quietly(api<Find[]>(paths.missionFinds(missionId)));
  if (finds) m.setFinds(finds);
  await loadChat(missionId);
  const track = await quietly(api<TrackFeature>(paths.missionTrack(missionId)));
  const coords = track?.geometry?.coordinates;
  if (coords) {
    m.setTrack(
      coords.flatMap((c) =>
        typeof c[0] === "number" && typeof c[1] === "number"
          ? [[c[0], c[1]] as [number, number]]
          : [],
      ),
    );
  }
}

export async function loadSectors(missionId: string): Promise<void> {
  const fc = await quietly(api<{ features?: SectorFeature[] }>(paths.missionSectors(missionId)));
  if (fc?.features && useMissionStore.getState().missionId === missionId)
    useMissionStore.getState().setSectors(fc.features);
}

/** First page, or after a reconnect everything after the newest message we have. */
async function loadChat(missionId: string): Promise<void> {
  const chat = useChatStore.getState();
  const after = lastChatSeq(chat.messages);
  if (after === null) {
    const page = await quietly(api<ChatPage>(paths.missionChatPage(missionId, { limit: 50 })));
    if (page) chat.receivePage(page.items, page.next_before);
    return;
  }
  for (let from = after, i = 0; i < 20; i += 1) {
    const page = await quietly(
      api<ChatPage>(paths.missionChatPage(missionId, { after_seq: from, limit: 50 })),
    );
    if (!page || page.items.length === 0) return;
    chat.receivePage(page.items);
    from = Math.max(from, ...page.items.map((x) => x.seq));
  }
}

/** Older messages when the operator scrolls up. */
export async function loadOlderChat(missionId: string): Promise<void> {
  const chat = useChatStore.getState();
  if (chat.nextBefore === null) return;
  const page = await quietly(
    api<ChatPage>(paths.missionChatPage(missionId, { before_seq: chat.nextBefore, limit: 50 })),
  );
  if (page) chat.receivePage(page.items, page.next_before);
}

/** Fetch an approval card's subject (charter or command) once, when its live event came
 * before the subject itself. */
export async function fetchSubject(a: Approval): Promise<void> {
  const m = useMissionStore.getState();
  if (m.subjectFetched[a.id]) return;
  m.markSubjectFetched(a.id);
  if (a.kind === "charter") {
    const list = await quietly(api<Charter[]>(paths.missionCharters(a.mission_id)));
    if (list) for (const c of list) useMissionStore.getState().upsertCharter(c);
  } else if (a.kind === "command") {
    const list = await quietly(
      api<Command[]>(paths.missionCommands(a.mission_id, "pending_approval")),
    );
    if (list) useMissionStore.getState().setCommands(list);
  }
}

function clearanceFrom(d: unknown): ClearanceView {
  const c = (d ?? {}) as { status?: string; valid_until?: string | null };
  const status = c.status;
  if (
    status === "active" ||
    status === "consumed" ||
    status === "expired" ||
    status === "invalidated"
  )
    return { status, valid_until: c.valid_until ?? null };
  return { status: "none", valid_until: null };
}

/** The latest find-mode plan's candidate, so the log line after reconnect can say "K<n>". */
let lastFind: { keepout_id: string | null; seq: number | null } | null = null;

/** An alert in the panel's own words (with its details) when sv.json has it, else the
 * ground's message_sv. Red ones stay until acknowledged; the link-down ones only go in the
 * log, since the banner already shows them. */
function showAlert(a: { code?: string; severity?: string; message_sv?: string }): void {
  const ui = useUiStore.getState();
  const view = a.code ? alertView(a.code, a as Record<string, unknown>, Date.now()) : null;
  if (!view) {
    ui.log(a.severity === "critical" ? "critical" : "info", a.message_sv ?? a.code ?? "");
    return;
  }
  const text = i18next.t(view.key, view.vars);
  if (isSticky(view)) ui.alarm(text);
  else {
    ui.log("info", text);
    if (!LINK_DOWN_CODES.has(a.code ?? "")) ui.toast(text);
  }
}

function logFindHoldEnded(d: Record<string, unknown>, ts: string | null | undefined): void {
  const at = (ts ? Date.parse(ts) : Number.NaN) || Date.now();
  const seq = lastFind && lastFind.keepout_id === d.keepout_id ? lastFind.seq : null;
  const v = findHoldEndedView(d, at, seq);
  const reason = v.reasonKey ? i18next.t(v.reasonKey) : String(d.reason ?? "");
  useUiStore.getState().log("info", i18next.t(v.key, { ...v.vars, reason }));
}

function isPending(a: Approval): boolean {
  return a.status === "pending" && a.kind !== "resume";
}

function apply(e: LiveEnvelope, vehicleId: string): void {
  const live = useLiveStore.getState();
  const m = useMissionStore.getState();
  const missionId = live.mission?.id ?? null;
  const forMission = e.mission_id == null || e.mission_id === missionId;
  live.patch({ lastWsRxAt: Date.now() });
  noteLive(e.type === "telemetry" ? (e.data as Telemetry).t : undefined);
  // Dispatch on the type: the server's topic names are not part of what the panel relies on.
  switch (e.type) {
    case "telemetry": {
      const t = e.data as Telemetry;
      live.patch({ telemetry: t, lastTelemetryAt: Date.now() });
      if (missionId && t.armed && Number.isFinite(t.lat) && Number.isFinite(t.lon))
        m.appendTrack(t.lon, t.lat);
      break;
    }
    case "vehicle.state": {
      const v = e.data as VehicleLive;
      const plan = v.link_loss_plan;
      if (plan?.mode === "find")
        lastFind = { keepout_id: plan.keepout_id ?? null, seq: plan.candidate_seq ?? null };
      live.patch({ vehicle: v });
      break;
    }
    case "alarm.SDLINK_LOST": {
      // On the vehicle topic, often before the plan: the plan's times fill in what is missing.
      const a = { ...live.vehicle?.link_loss_plan, ...(e.data as object), code: "SDLINK_LOST" };
      if ("lost_at_utc" in a) showAlert(a);
      break;
    }
    case "executor.find_hold_ended":
      logFindHoldEnded(e.data as Record<string, unknown>, e.ts);
      break;
    case "vehicle.action":
      actionBus.publish(e.data as ActionOutcome);
      break;
    case "latch.updated": {
      const d = e.data as Partial<OperatorLatch> & { active?: boolean };
      live.patch({ latch: d.active === false || !d.action ? null : (d as OperatorLatch) });
      break;
    }
    case "system.status": {
      const s = e.data as SystemStatus;
      live.patch({
        system: s,
        aiDisconnected: s.parts.brain?.state === "ok" ? false : live.aiDisconnected,
      });
      break;
    }
    case "mission.updated": {
      const mission = e.data as Mission;
      if (mission.id === missionId) live.patch({ mission });
      break;
    }
    case "mission.state_changed": {
      const d = e.data as { to?: Mission["status"]; created?: boolean };
      if (d.created || d.to === "CLOSED" || e.mission_id !== missionId) {
        void reloadMission(vehicleId);
      } else if (live.mission && d.to) {
        live.patch({ mission: { ...live.mission, status: d.to } });
      }
      break;
    }
    case "charter.updated":
      if (forMission) m.upsertCharter(e.data as Charter);
      break;
    case "approval.requested":
    case "approval.updated": {
      const a = e.data as Approval;
      if (!forMission || a.mission_id !== missionId) break;
      const isNew = !m.approvals[a.id];
      m.upsertApproval(a);
      if (isNew && isPending(a)) {
        useUiStore.getState().approvalRequested(a.id);
        void fetchSubjectIfMissing(a);
      }
      break;
    }
    case "command.created":
    case "command.updated": {
      const c = e.data as Command & { reason_codes?: string[]; message_sv?: string };
      if (!forMission) break;
      m.upsertCommand(c);
      if (c.status === "rejected_by_gate" && c.message_sv) {
        useUiStore.getState().log("info", `Grinden stoppade ${c.skill}: ${c.message_sv}`);
        // A drawn path the gate will not fly (outside the area, the battery): tell the pilot.
        if (c.skill === "follow_path")
          useUiStore.getState().toast(`Vägen nekades: ${c.message_sv}`);
      }
      break;
    }
    case "clearance.updated":
      if (forMission) m.setClearance(clearanceFrom(e.data));
      break;
    case "sector.updated": {
      const d = e.data as { sector_id?: string; status?: string; progress?: number };
      m.setSectors(
        m.sectors.map((f) =>
          f.properties.sector_id === d.sector_id
            ? {
                ...f,
                properties: {
                  ...f.properties,
                  status: d.status ?? f.properties.status,
                  progress: d.progress ?? f.properties.progress,
                },
              }
            : f,
        ),
      );
      break;
    }
    case "annotation.created":
      if (forMission) m.upsertAnnotation(e.data as Annotation);
      break;
    case "find.new":
    case "find.updated":
      if (forMission) m.upsertFind(e.data as Find);
      break;
    case "annotation.deleted":
      m.removeAnnotation((e.data as { id: string }).id);
      break;
    case "chat.message": {
      const msg = e.data as ChatMessage;
      if (msg.mission_id === missionId) useChatStore.getState().receive(msg);
      break;
    }
    case "chat.brain_status": {
      const { state, engine } = e.data as { state?: string; engine?: string | null };
      const busy = state === "busy";
      const known = state === "ready" || busy || state === "degraded" || state === "down";
      live.patch({
        brainBusy: busy,
        brainState: known ? state : live.brainState,
        brainEngine: engine === "cloud" || engine === "command" ? engine : null,
        brainBusySince: busy ? (live.brainBusySince ?? Date.now()) : null,
        aiDisconnected: state === "ready" || busy ? false : live.aiDisconnected,
      });
      break;
    }
    case "gate.decision": {
      const d = e.data as { decision?: string; message_sv?: string };
      if (d.decision === "REJECT") useUiStore.getState().log("info", d.message_sv ?? "");
      break;
    }
    case "alert": {
      const a = e.data as { severity?: string; message_sv?: string; code?: string };
      if (a.code === "BRAIN_DISCONNECTED") live.patch({ aiDisconnected: true });
      showAlert(a);
      break;
    }
    default:
      break; // unknown types are ignored
  }
}

async function fetchSubjectIfMissing(a: Approval): Promise<void> {
  const m = useMissionStore.getState();
  const have = a.kind === "charter" ? m.charters[a.subject_id] : m.commands[a.subject_id];
  if (!have) await fetchSubject(a);
}

let resyncing = false;
let held: LiveEnvelope[] = [];

async function reloadMission(vehicleId: string): Promise<void> {
  const mission = await loadMission(vehicleId);
  if (mission) await loadMissionData(mission.id);
}

/** Snapshots over REST; events that arrive meanwhile are applied afterwards, in order. */
export async function resync(vehicleId: string): Promise<void> {
  if (resyncing) return;
  resyncing = true;
  try {
    const v = await quietly(api<VehicleLive>(paths.vehicleState(vehicleId)));
    if (v) {
      useLiveStore
        .getState()
        .patch({ vehicle: v, telemetry: v.telemetry ?? useLiveStore.getState().telemetry });
    }
    await reloadMission(vehicleId);
  } finally {
    resyncing = false;
    const later = held.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
    held = [];
    for (const e of later) apply(e, vehicleId);
  }
}

/** Opens the live socket; every new server epoch (the first hello too) runs a resync. */
export function startLive(vehicleId: string): () => void {
  let firstEpoch = true;
  // Start loading at once: a socket that cannot connect must not leave the chat empty.
  const initial = resync(vehicleId);
  const sock = new LiveSocket(
    { vehicleId, missionId: () => useLiveStore.getState().mission?.id ?? null },
    {
      onState: (s) => useLiveStore.getState().setWs(s),
      onEnvelope: (e) => {
        // A STOP / FLY HOME outcome is not snapshot state: it goes on at once, so
        // "Kvitterat" never waits for a resync.
        if (resyncing && e.seq !== null && e.type !== "vehicle.action") held.push(e);
        else apply(e, vehicleId);
      },
      onNewEpoch: (_epoch, reason) => {
        const first = firstEpoch;
        firstEpoch = false;
        if (first && reason === "hello") return void initial;
        // A new epoch or resync_required: what we hold may be stale; start from nothing.
        useLiveStore.getState().reset();
        useMissionStore.getState().reset(null);
        useChatStore.getState().clear();
        void resync(vehicleId);
      },
    },
  );
  socket = sock;
  sock.start();
  const sectorPoll = setInterval(() => {
    const mission = useLiveStore.getState().mission;
    if (mission && FLYING.has(mission.status)) void loadSectors(mission.id);
  }, SECTOR_POLL_MS);
  return () => {
    clearInterval(sectorPoll);
    sock.stop();
    if (socket === sock) socket = null;
  };
}
