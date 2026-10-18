import { Monitor, Moon, MoreVertical, Sun } from "lucide-react";
import { useTranslation } from "react-i18next";
import { logout } from "../auth/bootstrap";
import { useNow } from "../lib/useNow";
import { useBrainState } from "../state/brain";
import { useLiveStore } from "../state/liveStore";
import { useMissionStore } from "../state/missionStore";
import { useSessionStore } from "../state/sessionStore";
import { useShellStore } from "../state/shellStore";
import { useUiStore } from "../state/uiStore";
import { THEME_PREFS, type ThemePref } from "../theme/theme";
import { type Cell, instrumentCells, isStale } from "./instruments";
import { Menu } from "./Menu";
import { type Tone, vehiclePill } from "./pills";

const clock = new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" });

function Bars({ n }: { n: number }) {
  return (
    <span className="bars" aria-hidden="true">
      {[1, 2, 3, 4].map((i) => (
        <span key={i} data-on={i <= n} style={{ height: `${3 + i * 2}px` }} />
      ))}
    </span>
  );
}

function InstrumentCell({ cell }: { cell: Cell }) {
  const { t } = useTranslation();
  const hint = cell.hintKey ? t(cell.hintKey) : undefined;
  const value = cell.id === "age" ? t("strip.ageValue", { v: cell.value }) : cell.value;
  return (
    <div
      className="cell"
      data-cell={cell.id}
      data-tone={cell.tone}
      data-testid={`cell-${cell.id}`}
      title={hint}
    >
      <span className="cell-label">{t(cell.labelKey)}</span>
      <span className="cell-value">
        {cell.bars !== undefined ? <Bars n={cell.bars} /> : null}
        {value || (cell.valueKey ? t(cell.valueKey) : null)}
        {cell.bar ? (
          <span className="battery-bar" aria-hidden="true">
            <span style={{ width: `${cell.bar.pct}%` }} />
            <i style={{ left: `${cell.bar.rtlPct}%` }} />
          </span>
        ) : null}
      </span>
      {hint && value ? <span className="sr-only">{hint}</span> : null}
    </div>
  );
}

/** The instrument cells: battery, time left, height, speed, heading, range, link and GPS. */
function Instruments() {
  const now = useNow(500);
  const telemetry = useLiveStore((s) => s.telemetry);
  const at = useLiveStore((s) => s.lastTelemetryAt);
  const link = useLiveStore((s) => s.vehicle?.link ?? null);
  const missionRtl = useLiveStore((s) => s.mission?.limits.battery_rtl_pct ?? null);
  const defaultRtl = useSessionStore((s) => s.config?.limits_default.battery_rtl_pct ?? 15);
  const cells = instrumentCells({
    telemetry,
    ageMs: at === null ? null : now - at,
    link,
    rtlPct: missionRtl ?? defaultRtl,
  });
  return (
    <div className="cells" data-testid="instruments">
      {cells.map((c) => (
        <InstrumentCell key={c.id} cell={c} />
      ))}
    </div>
  );
}

/** What the drone is doing; "Ingen kontakt" once its data is old, never a stale "Söker". */
function VehicleChip() {
  const { t } = useTranslation();
  const now = useNow(500);
  const snap = useLiveStore((s) => s.vehicle?.snapshot);
  const at = useLiveStore((s) => s.lastTelemetryAt);
  const ws = useLiveStore((s) => s.ws);
  const lost = ws === "down" || (at !== null && isStale(now - at));
  const vp = vehiclePill(snap);
  const tone: Tone = lost ? "yellow" : vp.tone;
  return (
    <span className="state-chip" data-tone={tone} data-testid="pill-vehicle">
      <span className="dot" aria-hidden="true" />
      {lost ? t("pills.noContact") : (vp.raw ?? t(vp.key))}
    </span>
  );
}

/** The autonomy mode in words, the flight controller's mode in small capitals. */
function ModeCell() {
  const { t } = useTranslation();
  const mission = useLiveStore((s) => s.mission);
  const fcMode = useLiveStore((s) => s.telemetry?.flight_mode ?? null);
  const words = mission?.autonomy_mode
    ? t(mission.autonomy_mode === "WITHIN_CHARTER" ? "strip.modeB" : "strip.modeA")
    : "–";
  return (
    <div className="cell cell-mode" data-cell="mode" data-testid="pill-mode">
      <span className="cell-label">{t("strip.mode")}</span>
      <span className="cell-value">
        {words}
        {fcMode ? <small className="fc-mode">{fcMode}</small> : null}
      </span>
    </div>
  );
}

function Clock() {
  const { t } = useTranslation();
  const now = useNow(1_000);
  return (
    <div className="cell cell-clock" data-cell="clock">
      <span className="cell-label">{t("strip.clock")}</span>
      <span className="cell-value">{clock.format(now)}</span>
    </div>
  );
}

/** The AI's state in words, beside the chat's heading. */
export function AiStatus() {
  const { t } = useTranslation();
  const brain = useBrainState();
  const busy = useLiveStore((s) => s.brainBusy);
  const ai =
    brain === "down"
      ? "pills.ai.down"
      : busy
        ? "pills.ai.busy"
        : brain === "ok"
          ? "pills.ai.ready"
          : "pills.ai.unknown";
  const tone =
    brain === "down" ? "bad" : brain === "degraded" ? "warn" : brain === "ok" ? "ok" : "mute";
  return (
    <span className="ai-status" data-tone={tone} data-testid="pill-ai">
      <span className="dot" aria-hidden="true" />
      <span className="ai-status-text">{t(ai)}</span>
    </span>
  );
}

/** Pending approval cards, also when the chat is folded away; pressing it shows the card. */
function PendingBadge() {
  const { t } = useTranslation();
  const approvals = useMissionStore((s) => s.approvals);
  const pending = Object.values(approvals).filter(
    (a) => a.status === "pending" && a.kind !== "resume",
  );
  if (pending.length === 0) return null;
  return (
    <button
      type="button"
      className="pending-badge"
      data-testid="pending-badge"
      onClick={() => {
        useShellStore.getState().showTab("chat");
        useUiStore.getState().showApproval(pending[0]?.id ?? null);
      }}
    >
      <span className="dot" aria-hidden="true" />
      {t("strip.waiting", { n: pending.length })}
    </button>
  );
}

const THEME_ICON: Record<ThemePref, typeof Sun> = { auto: Monitor, dark: Moon, light: Sun };

/** Auto, dark and daylight, each one press away. */
function ThemeSwitch() {
  const { t } = useTranslation();
  const pref = useUiStore((s) => s.themePref);
  return (
    <fieldset className="seg seg-icons">
      <legend className="sr-only">{t("theme.label")}</legend>
      {THEME_PREFS.map((p) => {
        const Icon = THEME_ICON[p];
        return (
          <button
            key={p}
            type="button"
            aria-pressed={pref === p}
            aria-label={t(`theme.${p}`)}
            title={t(`theme.${p}`)}
            data-testid={`theme-${p}`}
            onClick={() => useUiStore.getState().setThemePref(p)}
          >
            <Icon aria-hidden="true" size={15} />
          </button>
        );
      })}
    </fieldset>
  );
}

/** Flyg (live) and Granska (finds review). */
function ModeSwitch() {
  const { t } = useTranslation();
  const mode = useShellStore((s) => s.mode);
  return (
    <div className="seg" role="tablist" aria-label={t("strip.view")}>
      {(["fly", "review"] as const).map((m) => (
        <button
          key={m}
          type="button"
          role="tab"
          aria-selected={mode === m}
          data-testid={`mode-${m}`}
          onClick={() => useShellStore.getState().setMode(m)}
        >
          {t(`strip.${m}`)}
        </button>
      ))}
    </div>
  );
}

export function TopBar() {
  const { t } = useTranslation();
  const title = useLiveStore((s) => s.mission?.title ?? null);
  const sim = useSessionStore((s) => s.config?.vehicle_kind === "sim");
  const vp = useShellStore((s) => s.vp);
  return (
    <header className="topbar" data-testid="topbar">
      <div className="brand-block">
        <h1 className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">{t("app.name")}</span>
        </h1>
        <span className="mission-title" data-testid="mission-title" title={title ?? undefined}>
          {title ?? t("app.noMission")}
        </span>
        {sim ? <span className="tag">{t("app.sim")}</span> : null}
      </div>
      {vp !== "phone" ? <ModeSwitch /> : null}
      <VehicleChip />
      {vp !== "phone" ? <Instruments /> : null}
      {vp === "desktop" ? <ModeCell /> : null}
      {vp === "desktop" ? <Clock /> : null}
      <div className="topbar-tools">
        <PendingBadge />
        {vp !== "phone" ? <ThemeSwitch /> : null}
        <Menu
          label={t("menu.label")}
          icon={<MoreVertical aria-hidden="true" size={18} />}
          testId="main-menu"
        >
          {(close) => (
            <>
              {vp === "phone" ? (
                <div className="menu-row">
                  <ThemeSwitch />
                </div>
              ) : null}
              <button
                type="button"
                className="menu-item"
                data-testid="menu-status"
                onClick={() => {
                  useShellStore.getState().setOverlay("status");
                  close();
                }}
              >
                {t("menu.status")}
              </button>
              <button
                type="button"
                className="menu-item"
                data-testid="menu-manual"
                onClick={() => {
                  useShellStore.getState().setOverlay("manual");
                  close();
                }}
              >
                {t("menu.manual")}
              </button>
              <button
                type="button"
                className="menu-item"
                onClick={() => {
                  close();
                  void logout();
                }}
              >
                {t("menu.logout")}
              </button>
            </>
          )}
        </Menu>
      </div>
    </header>
  );
}

/** The phone's instrument row under the top bar (the bar itself has no room). */
export function PhoneInstruments() {
  const { t } = useTranslation();
  return (
    <section className="phone-strip" aria-label={t("strip.label")}>
      <Instruments />
    </section>
  );
}
