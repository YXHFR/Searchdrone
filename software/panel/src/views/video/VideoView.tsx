import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../../api/http";
import { paths } from "../../api/paths";
import type { SystemStatus } from "../../api/types";
import { noteVideoFps } from "../../lib/testHooks";
import { useSessionStore } from "../../state/sessionStore";
import type { ViewProps } from "../registry";
import { startWhep, type WhepSession } from "./whepClient";

/** No playing picture within this time over WebRTC: switch to HLS. */
export const WHEP_PLAY_MS = 5000;
/** No picture at all within this time: show "no video" and retry this often. */
export const NO_VIDEO_MS = 10_000;
const POLL_MS = 250;

type Mode = "connecting" | "webrtc" | "hls" | "none";

interface Info {
  width: number;
  height: number;
  fps: number | null;
}

/** Playing = there is a frame (readyState >= 2) and the time moves on. */
function playing(el: HTMLVideoElement, last: { t: number }): boolean {
  const moved = el.currentTime > last.t;
  last.t = el.currentTime;
  return el.readyState >= 2 && moved;
}

async function videoServerDown(): Promise<boolean> {
  try {
    const status = await api<SystemStatus>(paths.systemStatus());
    return status.parts.mediamtx?.state === "down";
  } catch {
    return false; // unknown: try the video anyway
  }
}

/**
 * HLS where the browser plays it by itself (Safari, every iPhone browser): the case the
 * fallback is for. Elsewhere it is not attempted and the view ends at "no video".
 */
function startHls(el: HTMLVideoElement, url: string): boolean {
  if (!el.canPlayType("application/vnd.apple.mpegurl")) return false;
  el.src = url;
  void el.play().catch(() => undefined);
  return true;
}

/**
 * One attempt at the live picture: WHEP first, HLS when WebRTC gives no picture within
 * 5 s, "no video" after 10 s. Returns a function that stops everything (and ends the
 * WHEP session).
 */
function connect(
  el: HTMLVideoElement,
  whepUrl: string,
  hlsUrl: string | null,
  onMode: (m: Mode) => void,
): () => void {
  let stopped = false;
  let whep: WhepSession | null = null;
  const abort = new AbortController();
  const timers: ReturnType<typeof setTimeout>[] = [];
  const last = { t: -1 };
  let mode: Mode = "connecting";
  const set = (m: Mode) => {
    mode = m;
    onMode(m);
  };

  const dropWhep = () => {
    const s = whep;
    whep = null;
    if (s) void s.close();
    el.srcObject = null;
  };

  const toHls = () => {
    if (stopped || mode === "hls" || mode === "none") return;
    dropWhep();
    if (!hlsUrl) return;
    set("hls");
    startHls(el, hlsUrl); // without native HLS the no-video timer ends the attempt
  };

  const tick = setInterval(() => {
    if (stopped || mode === "none") return;
    if (playing(el, last) && mode === "connecting") set("webrtc");
  }, POLL_MS);

  timers.push(
    setTimeout(() => {
      if (mode === "connecting") toHls();
    }, WHEP_PLAY_MS),
    setTimeout(() => {
      if (stopped || mode === "webrtc" || (mode === "hls" && el.readyState >= 2)) return;
      dropWhep();
      set("none");
    }, NO_VIDEO_MS),
  );

  void startWhep(whepUrl, abort.signal)
    .then((s) => {
      if (stopped) {
        void s.close();
        return;
      }
      whep = s;
      el.srcObject = s.stream;
      void el.play().catch(() => undefined);
    })
    .catch(() => toHls());

  return () => {
    stopped = true;
    abort.abort();
    clearInterval(tick);
    for (const t of timers) clearTimeout(t);
    dropWhep();
    el.removeAttribute("src");
    el.load();
  };
}

export default function VideoView(_: ViewProps) {
  const { t } = useTranslation();
  const video = useSessionStore((s) => s.config?.video ?? null);
  const ref = useRef<HTMLVideoElement>(null);
  const [mode, setMode] = useState<Mode>(video ? "connecting" : "none");
  const [attempt, setAttempt] = useState(0);
  const [visible, setVisible] = useState(() => document.visibilityState !== "hidden");
  const [info, setInfo] = useState<Info | null>(null);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  useEffect(() => {
    const onVis = () => setVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: attempt restarts the stream
  useEffect(() => {
    const el = ref.current;
    if (!video || !el || !visible) {
      setMode("none");
      return;
    }
    let stop: (() => void) | null = null;
    let cancelled = false;
    setMode("connecting");
    void videoServerDown().then((down) => {
      if (cancelled) return;
      if (down) setMode("none");
      else stop = connect(el, video.whep_url, video.hls_url, setMode);
    });
    return () => {
      cancelled = true;
      stop?.();
    };
  }, [video, visible, attempt]);

  // "No video": try again every 10 s by itself.
  useEffect(() => {
    if (mode !== "none" || !video) return;
    const timer = setTimeout(retry, NO_VIDEO_MS);
    return () => clearTimeout(timer);
  }, [mode, video, retry]);

  // Resolution and frame rate for the corner badge.
  useEffect(() => {
    if (mode !== "webrtc" && mode !== "hls") {
      setInfo(null);
      noteVideoFps(null);
      return;
    }
    let frames = ref.current?.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
    const timer = setInterval(() => {
      const el = ref.current;
      if (!el) return;
      const now = el.getVideoPlaybackQuality?.().totalVideoFrames;
      const fps = now === undefined ? null : now - frames;
      frames = now ?? frames;
      noteVideoFps(fps);
      setInfo({ width: el.videoWidth, height: el.videoHeight, fps });
    }, 1000);
    return () => clearInterval(timer);
  }, [mode]);

  const badge =
    mode === "webrtc" ? t("views.video.live") : mode === "hls" ? t("views.video.hls") : null;
  return (
    <div className="view-body video-body" data-testid="video-view" data-mode={mode}>
      <video
        ref={ref}
        className="video-el"
        data-testid="video-element"
        muted
        playsInline
        autoPlay
        controls={false}
        disablePictureInPicture
        hidden={mode === "none"}
      />
      {badge && (
        <div className="video-badge" data-testid="video-badge">
          <span>{badge}</span>
          {info && info.width > 0 && (
            <span>
              {info.width}×{info.height}
              {info.fps !== null ? ` · ${info.fps} fps` : ""}
            </span>
          )}
        </div>
      )}
      {mode === "connecting" && <p className="video-note">{t("views.video.connecting")}</p>}
      {mode === "none" && (
        <div className="view-empty video-empty video-none" data-testid="video-none">
          <p>{video ? t("views.video.noPicture") : t("views.video.none")}</p>
          <button type="button" className="btn" data-testid="video-retry" onClick={retry}>
            {t("views.video.retry")}
          </button>
        </div>
      )}
    </div>
  );
}
