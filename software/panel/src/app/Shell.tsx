import { ArrowLeft, ArrowLeftRight, VideoOff, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { startLive } from "../api/live";
import { ReauthSheet } from "../auth/ReauthSheet";
import { Toasts } from "../header/Toasts";
import { PhoneInstruments, TopBar } from "../header/TopBar";
import { mmss, useNow } from "../lib/useNow";
import { FindsPanel, MissionPanel } from "../mission/MissionPanel";
import { linkDownBanner } from "../safety/linkDown";
import { SideBody, SideTabs } from "../side/SidePanel";
import { useLiveStore } from "../state/liveStore";
import { useSessionStore } from "../state/sessionStore";
import { type SheetSize, useShellStore, viewportFor } from "../state/shellStore";
import { ViewSlot } from "./ViewSlot";

/** The ground station has been unreachable this long before the banner shows. */
const BANNER_AFTER_MS = 2_000;

function useViewport(): void {
  useEffect(() => {
    const update = (): void => useShellStore.getState().setViewport(viewportFor(window.innerWidth));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
}

/** One banner over the stage, never pushing the layout down. No ground station goes first;
 * else the drone's link loss with what it is expected to do, counting down. */
export function StageBanner() {
  const { t } = useTranslation();
  const since = useLiveStore((s) => s.wsDownSince);
  const ws = useLiveStore((s) => s.ws);
  const vehicle = useLiveStore((s) => s.vehicle);
  const now = useNow(250);
  if (since !== null) {
    if (now - since < BANNER_AFTER_MS) return null;
    return (
      <div className="stage-banner" role="status" data-testid="connection-banner">
        {t("banner.noGround")}
      </div>
    );
  }
  const down = ws === "up" ? linkDownBanner(vehicle, now) : null;
  if (!down) return null;
  return (
    <div
      className="stage-banner"
      role="status"
      data-tone={down.tone}
      data-code={down.code}
      data-testid="link-down-banner"
    >
      {t(down.key, down.vars)}
      {down.next ? (
        <span className="stage-banner-next num">
          {t(down.next.key, { left: mmss(down.next.leftS * 1000) })}
        </span>
      ) : null}
    </div>
  );
}

/** Map and video share the stage: one fills it, the other is a small window to swap with. */
function Stage({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const main = useShellStore((s) => s.main);
  const hasVideo = useSessionStore((s) => s.config?.video != null);
  const mapMain = main === "map" || !hasVideo;
  const swap = (to: "map" | "video") => (
    <button
      type="button"
      className="pip-swap"
      data-testid={`pip-swap-${to}`}
      onClick={() => useShellStore.getState().setMain(to)}
    >
      <ArrowLeftRight aria-hidden="true" size={13} />
      {t(to === "map" ? "stage.bigMap" : "stage.bigVideo")}
    </button>
  );
  return (
    <div className="stage" data-main={mapMain ? "map" : "video"}>
      <section
        className={`surface ${mapMain ? "is-main" : "is-pip"}`}
        aria-label={t("views.map.title")}
        data-testid="view-map"
      >
        <ViewSlot view="map" vehicleId={vehicleId} isCompact={!mapMain} />
        {mapMain ? null : swap("map")}
      </section>
      {hasVideo ? (
        <section
          className={`surface ${mapMain ? "is-pip" : "is-main"}`}
          aria-label={t("views.video.title")}
          data-testid="view-video"
        >
          <ViewSlot view="video" vehicleId={vehicleId} isCompact={mapMain} />
          {mapMain ? swap("video") : null}
        </section>
      ) : (
        <span className="no-camera" data-testid="no-camera">
          <VideoOff aria-hidden="true" size={13} /> {t("stage.noCamera")}
        </span>
      )}
    </div>
  );
}

/** Granska comes with the candidate queue; the chat stays beside it. */
function ReviewStage({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const phone = useShellStore((s) => s.vp === "phone");
  return (
    <section className="review-stage" data-testid="view-candidates">
      {phone ? (
        <button
          type="button"
          className="btn btn-quiet"
          data-testid="review-back"
          onClick={() => useShellStore.getState().setMode("fly")}
        >
          <ArrowLeft aria-hidden="true" size={15} /> {t("stage.backToFly")}
        </button>
      ) : null}
      <ViewSlot view="candidates" vehicleId={vehicleId} isCompact={false} />
    </section>
  );
}

/** Systemstatus and Styr själv cover the stage until closed; the columns stay. */
function Overlay({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const overlay = useShellStore((s) => s.overlay);
  if (!overlay) return null;
  return (
    <section className="overlay" aria-labelledby="overlay-title" data-testid={`view-${overlay}`}>
      <header className="overlay-head">
        <h2 id="overlay-title">{t(`views.${overlay}.title`)}</h2>
        <button
          type="button"
          className="btn btn-quiet"
          data-testid="overlay-close"
          onClick={() => useShellStore.getState().setOverlay(null)}
        >
          <X aria-hidden="true" size={16} /> {t("common.close")}
        </button>
      </header>
      {/* Scrolls on a small screen; focusable so the keyboard can scroll it too. */}
      {/* biome-ignore lint/a11y/noNoninteractiveTabindex: a scrolling region must take focus */}
      <div className="overlay-body" tabIndex={0}>
        <ViewSlot view={overlay} vehicleId={vehicleId} isCompact={false} />
      </div>
    </section>
  );
}

const SIZES: SheetSize[] = ["peek", "half", "full"];

/** Phone: chat and events in a sheet over the map. Drag the handle, or tap it to step up. */
function Sheet({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const size = useShellStore((s) => s.sheet);
  const ref = useRef<HTMLElement>(null);
  const drag = useRef<{ y: number; h: number; moved: boolean } | null>(null);
  const end = (): void => {
    const d = drag.current;
    const el = ref.current;
    drag.current = null;
    if (!d || !el) return;
    const shell = useShellStore.getState();
    if (!d.moved) {
      shell.setSheet(SIZES[(SIZES.indexOf(size) + 1) % SIZES.length] ?? "peek");
      return;
    }
    const frac = el.offsetHeight / (el.parentElement?.offsetHeight ?? window.innerHeight);
    el.style.height = "";
    shell.setSheet(frac < 0.33 ? "peek" : frac < 0.75 ? "half" : "full");
  };
  return (
    <section
      ref={ref}
      className="sheet"
      data-size={size}
      aria-label={t("side.label")}
      data-testid="sheet"
    >
      <button
        type="button"
        className="sheet-handle"
        data-testid="sheet-handle"
        aria-label={t(size === "full" ? "stage.sheetDown" : "stage.sheetUp")}
        onPointerDown={(e) => {
          const el = ref.current;
          if (!el) return;
          drag.current = { y: e.clientY, h: el.offsetHeight, moved: false };
          e.currentTarget.setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          const el = ref.current;
          if (!d || !el) return;
          const dy = d.y - e.clientY;
          if (Math.abs(dy) > 6) d.moved = true;
          if (d.moved) el.style.height = `${Math.max(120, d.h + dy)}px`;
        }}
        onPointerUp={end}
        onPointerCancel={end}
      >
        <i aria-hidden="true" />
      </button>
      <SideTabs withMission />
      <SideBody vehicleId={vehicleId} withMission />
    </section>
  );
}

export function Shell({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  useViewport();
  const vp = useShellStore((s) => s.vp);
  const mode = useShellStore((s) => s.mode);
  const needsReauth = useSessionStore((s) => s.needsReauth);
  useEffect(() => startLive(vehicleId), [vehicleId]);
  return (
    <div className="shell" data-vp={vp} data-mode={mode}>
      <TopBar />
      {vp === "phone" ? <PhoneInstruments /> : null}
      <main className="shell-body" id="main">
        {vp === "desktop" ? (
          <aside className="left-col" aria-label={t("mission.column")}>
            <MissionPanel />
            <FindsPanel />
          </aside>
        ) : null}
        <div className="centre">
          {mode === "review" ? (
            <ReviewStage vehicleId={vehicleId} />
          ) : (
            <Stage vehicleId={vehicleId} />
          )}
          <StageBanner />
          <Overlay vehicleId={vehicleId} />
          {vp === "phone" ? <Sheet vehicleId={vehicleId} /> : null}
        </div>
        {vp !== "phone" ? (
          <aside className="side-col" aria-label={t("side.label")}>
            <SideTabs withMission={vp === "medium"} />
            <SideBody vehicleId={vehicleId} withMission={vp === "medium"} />
          </aside>
        ) : null}
      </main>
      <Toasts />
      {needsReauth ? <ReauthSheet /> : null}
    </div>
  );
}
