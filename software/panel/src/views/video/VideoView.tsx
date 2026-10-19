import { type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../../api/http";
import { paths } from "../../api/paths";
import type { SystemStatus } from "../../api/types";
import { noteVideoFps } from "../../lib/testHooks";
import { useNow } from "../../lib/useNow";
import { useSessionStore } from "../../state/sessionStore";
import type { ViewProps } from "../registry";
import { Hud } from "./Hud";
import { startWhep, type WhepSession } from "./whepClient";

/** No playing picture within this time over WebRTC: switch to HLS. */
export const WHEP_PLAY_MS = 5000;
/** No picture at all within this time: show "no video" and retry this often. */
export const NO_VIDEO_MS = 10_000;
/** A started picture that stops moving this long (or a lost WebRTC link): "no video". */
export const STALL_MS = 3000;
/** How often the last picture is kept, to show frozen and grey when the stream is lost. */
const SNAPSHOT_MS = 1000;
const POLL_MS = 250;

export type Mode = "connecting" | "webrtc" | "hls" | "none";

interface Info {
  width: number;
  height: number;
  fps: number | null;
}

/**
 * Playing = there is a frame (readyState >= 2) and the picture moves on: new decoded
 * frames where the browser counts them, else the time. A WebRTC element's time can keep
 * running after the frames stop (camera gone, link still up), so frames come first.
 */
function playing(el: HTMLVideoElement, last: { t: number; f: number }): boolean {
  const frames = el.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
  const moved = frames > 0 ? frames > last.f : el.currentTime > last.t;
  last.t = el.currentTime;
  last.f = frames;
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
 * 5 s, "no video" after 10 s. Once the picture plays, it ends at "no video" when the
 * picture stands still for 3 s or the WebRTC connection is lost. Returns a function that
 * stops everything (and ends the WHEP session).
 */
export function connect(
  el: HTMLVideoElement,
  whepUrl: string,
  hlsUrl: string | null,
  onMode: (m: Mode) => void,
): () => void {
  let stopped = false;
  let whep: WhepSession | null = null;
  const abort = new AbortController();
  const timers: ReturnType<typeof setTimeout>[] = [];
  const last = { t: -1, f: -1 };
  let started = false;
  let still = 0; // polls in a row without progress after the picture started
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

  const lose = () => {
    if (stopped || mode === "none") return;
    dropWhep();
    el.removeAttribute("src");
    el.load();
    set("none"); // the view tries again after NO_VIDEO_MS
  };

  const tick = setInterval(() => {
    if (stopped || mode === "none") return;
    if (playing(el, last)) {
      started = true;
      still = 0;
      if (mode === "connecting") set("webrtc");
    } else if (started && ++still * POLL_MS >= STALL_MS) {
      lose();
    }
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

  void startWhep(whepUrl, abort.signal, () => (mode === "connecting" ? toHls() : lose()))
    .then((s) => {
      if (stopped || mode !== "connecting") {
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

/** Keeps the last good frame in a canvas while the picture plays; returns when it was taken. */
function useLastFrame(
  video: RefObject<HTMLVideoElement | null>,
  canvas: RefObject<HTMLCanvasElement | null>,
  playing: boolean,
): number | null {
  const [at, setAt] = useState<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    const grab = (): void => {
      const v = video.current;
      const c = canvas.current;
      if (!v || !c || v.videoWidth === 0) return;
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext("2d")?.drawImage(v, 0, 0);
      setAt(Date.now());
    };
    grab();
    const timer = setInterval(grab, SNAPSHOT_MS);
    return () => clearInterval(timer);
  }, [playing, video, canvas]);
  return at;
}

export default function VideoView({ isCompact }: ViewProps) {
  const { t } = useTranslation();
  const video = useSessionStore((s) => s.config?.video ?? null);
  const ref = useRef<HTMLVideoElement>(null);
  const frame = useRef<HTMLCanvasElement>(null);
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

  // React sets `muted` only as a property. Autoplay policies and checkers such as axe read the
  // attribute; without it axe waits for the metadata of what looks like a video with sound.
  useEffect(() => {
    ref.current?.setAttribute("muted", "");
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

  const live = mode === "webrtc" || mode === "hls";
  const frameAt = useLastFrame(ref, frame, live);
  const now = useNow(1_000);
  const frozen = !live && frameAt !== null;
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
      <canvas ref={frame} className="video-frozen" data-testid="video-frozen" hidden={!frozen} />
      {frozen && frameAt !== null ? (
        <div className="video-badge video-badge-stale" data-testid="video-stale">
          {t("views.video.old", { s: Math.max(0, Math.round((now - frameAt) / 1000)) })}
        </div>
      ) : null}
      {live && !isCompact ? <Hud /> : null}
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
        <div
          className={`view-empty video-none${frozen ? " video-none-over" : ""}`}
          data-testid="video-none"
        >
          <p>{video ? t("views.video.noPicture") : t("views.video.none")}</p>
          <button type="button" className="btn" data-testid="video-retry" onClick={retry}>
            {t("views.video.retry")}
          </button>
        </div>
      )}
    </div>
  );
}
