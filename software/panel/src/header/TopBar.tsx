import { BellRing, LayoutGrid, MoreVertical } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { logout } from "../auth/bootstrap";
import { PRESET_IDS } from "../layout/layoutModel";
import { useBrainState } from "../state/brain";
import { useLayoutStore } from "../state/layoutStore";
import { useLiveStore } from "../state/liveStore";
import { useMissionStore } from "../state/missionStore";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { Menu } from "./Menu";
import { linkQuality, STALE_MS, type Tone, vehiclePill } from "./pills";

function useNow(ms: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}

function Pill({
  tone,
  children,
  testId,
}: {
  tone: Tone;
  children: React.ReactNode;
  testId?: string;
}) {
  return (
    <span className={`pill pill-${tone}`} data-testid={testId}>
      {children}
    </span>
  );
}

function StatusPills() {
  const { t } = useTranslation();
  const now = useNow(500);
  const tel = useLiveStore((s) => s.telemetry);
  const at = useLiveStore((s) => s.lastTelemetryAt);
  const vehicle = useLiveStore((s) => s.vehicle);
  const ws = useLiveStore((s) => s.ws);
  const mission = useLiveStore((s) => s.mission);
  const brain = useBrainState();
  const busy = useLiveStore((s) => s.brainBusy);
  const rtl = useSessionStore(
    (s) => mission?.limits.battery_rtl_pct ?? s.config?.limits_default.battery_rtl_pct ?? 15,
  );
  const age = at === null ? null : now - at;
  const stale = age === null || age > STALE_MS;
  const vp = vehiclePill(vehicle?.snapshot);
  const link = linkQuality(vehicle?.link, age);
  const battery = tel?.battery_pct ?? null;
  const batteryTone: Tone =
    stale || battery === null ? "grey" : battery <= rtl ? "red" : battery < 30 ? "yellow" : "green";
  const old = stale && tel ? ` ${t("pills.stale")}` : "";
  const ai =
    brain === "down"
      ? "pills.ai.down"
      : busy
        ? "pills.ai.busy"
        : brain === "ok"
          ? "pills.ai.ready"
          : "pills.ai.unknown";

  return (
    <div className="pills">
      <Pill tone={vp.tone} testId="pill-vehicle">
        {vp.raw ?? t(vp.key)}
      </Pill>
      <Pill tone={batteryTone} testId="pill-battery">
        {t("pills.battery", { v: battery === null ? "–" : Math.round(battery) })}
        {old}
      </Pill>
      <Pill tone={stale ? "grey" : "blue"} testId="pill-alt">
        {t("pills.alt", { v: tel ? Math.round(tel.alt_rel_home_m) : "–" })}
        {old}
      </Pill>
      <Pill
        tone={link === "good" ? "green" : link === "weak" ? "yellow" : "red"}
        testId="pill-link"
      >
        {t(`pills.link.${link}`)}
      </Pill>
      {mission?.autonomy_mode ? (
        <Pill tone="blue" testId="pill-mode">
          {t(mission.autonomy_mode === "WITHIN_CHARTER" ? "pills.mode.b" : "pills.mode.a")}
        </Pill>
      ) : null}
      <Pill tone={brain === "down" ? "red" : "grey"} testId="pill-ai">
        {t(ai)}
      </Pill>
      {ws === "down" ? (
        <Pill tone="red" testId="pill-panel">
          {t("pills.noContact")}
        </Pill>
      ) : null}
    </div>
  );
}

/** Pending approval cards, also when the chat is hidden; pressing it shows the chat. */
function PendingApprovalsBadge() {
  const { t } = useTranslation();
  const n = useMissionStore(
    (s) =>
      Object.values(s.approvals).filter((a) => a.status === "pending" && a.kind !== "resume")
        .length,
  );
  if (n === 0) return null;
  return (
    <button
      type="button"
      className="pill pill-yellow badge-btn"
      data-testid="pending-badge"
      aria-label={t("cards.pending", { n })}
      onClick={() => useLayoutStore.getState().showView("chat")}
    >
      <BellRing aria-hidden="true" size={14} />
      {n}
    </button>
  );
}

export function TopBar() {
  const { t } = useTranslation();
  const title = useLiveStore((s) => s.mission?.title ?? null);
  const sim = useSessionStore((s) => s.config?.vehicle_kind === "sim");
  const vp = useLayoutStore((s) => s.vp);
  const theme = useUiStore((s) => s.theme);
  return (
    <header className="topbar" data-testid="topbar">
      <h1 className="brand">{t("app.name")}</h1>
      {sim ? <span className="tag">{t("app.simulator")}</span> : null}
      <span className="mission-title" data-testid="mission-title">
        {title ?? t("app.noMission")}
      </span>
      <StatusPills />
      <PendingApprovalsBadge />
      <div className="topbar-tools">
        {vp !== "mobile" ? (
          <Menu
            label={t("layout.menu")}
            icon={<LayoutGrid aria-hidden="true" size={18} />}
            testId="layout-menu"
          >
            {(close) =>
              PRESET_IDS.map((id) => (
                <button
                  key={id}
                  type="button"
                  className="menu-item"
                  data-testid={`preset-${id}`}
                  onClick={() => {
                    useLayoutStore.getState().applyPreset(id);
                    close();
                  }}
                >
                  {t(`layout.preset.${id}`)}
                </button>
              ))
            }
          </Menu>
        ) : null}
        <Menu
          label={t("menu.label")}
          icon={<MoreVertical aria-hidden="true" size={18} />}
          testId="main-menu"
        >
          {(close) => (
            <>
              <button
                type="button"
                className="menu-item"
                onClick={() => {
                  useUiStore.getState().setTheme(theme === "dark" ? "light" : "dark");
                  close();
                }}
              >
                {t(theme === "dark" ? "menu.themeLight" : "menu.themeDark")}
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

const BANNER_AFTER_MS = 2_000;

export function ConnectionBanner() {
  const { t } = useTranslation();
  const since = useLiveStore((s) => s.wsDownSince);
  const now = useNow(250);
  if (since === null || now - since < BANNER_AFTER_MS) return null;
  return (
    <div className="banner banner-bad" role="status" data-testid="connection-banner">
      {t("banner.noGround")}
    </div>
  );
}
