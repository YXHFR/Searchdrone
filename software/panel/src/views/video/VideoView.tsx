import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLiveStore } from "../../state/liveStore";
import { useSessionStore } from "../../state/sessionStore";
import type { ViewProps } from "../registry";
import { startWhep, type WhepSession } from "./whepClient";

// WHEP first (lowest delay), HLS as fallback, then "no video" with a retry every 10 s.
const PLAYING_WITHIN_MS = 5000;
const GIVE_UP_MS = 10_000;
const RETRY_MS = 10_000;

type Mode = "connecting" | "webrtc" | "hls" | "none";

interface HlsLike {
  loadSource(url: string): void;
  attachMedia(el: HTMLMediaElement): void;
  destroy(): void;
}

/** Resolves when the element shows moving pictures, false after `ms`. */
function waitPlaying(el: HTMLVideoElement, ms: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    const start = Date.now();
    let last = el.currentTime;
    const tick = () => {
      if (signal.aborted) return resolve(false);
      if (el.readyState >= 2 && el.currentTime > last) return resolve(true);
      last = el.currentTime;
      if (Date.now() - start >= ms) return resolve(false);
      setTimeout(tick, 200);
    };
    setTimeout(tick, 200);
  });
}

async function startHls(el: HTMLVideoElement, url: string): Promise<() => void> {
  if (el.canPlayType("application/vnd.apple.mpegurl")) {
    el.src = url; // Safari: built-in HLS
    return () => el.removeAttribute("src");
  }
  const { default: Hls } = await import("hls.js");
  if (!Hls.isSupported()) throw new Error("HLS stöds inte");
  const hls: HlsLike = new Hls({ lowLatencyMode: true }); // same origin: the cookie goes along
  hls.loadSource(url);
  hls.attachMedia(el);
  return () => hls.destroy();
}

export default function VideoView(_: ViewProps) {
  const { t } = useTranslation();
  const video = useSessionStore((s) => s.config?.video ?? null);
  const serverDown = useLiveStore((s) => s.system?.parts.mediamtx?.state === "down");
  const el = useRef<HTMLVideoElement>(null);
  const [mode, setMode] = useState<Mode>("connecting");
  const [attempt, setAttempt] = useState(0);
  const [visible, setVisible] = useState(() => document.visibilityState !== "hidden");
  const [info, setInfo] = useState("");
  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  useEffect(() => {
    const onVis = () => setVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: a new attempt reconnects
  useEffect(() => {
    const media = el.current;
    if (!video || serverDown || !visible || !media) {
      setMode("none");
      return;
    }
    const abort = new AbortController();
    let whep: WhepSession | null = null;
    let stopHls: (() => void) | null = null;
    const started = Date.now();
    setMode("connecting");
    void (async () => {
      try {
        whep = await startWhep(video.whep_url, abort.signal);
        media.srcObject = whep.stream;
        void media.play().catch(() => undefined);
        if (await waitPlaying(media, PLAYING_WITHIN_MS, abort.signal)) {
          setMode("webrtc");
          return;
        }
      } catch {
        // falls through to HLS
      }
      if (abort.signal.aborted) return;
      whep?.close();
      whep = null;
      media.srcObject = null;
      if (video.hls_url) {
        setMode("hls");
        try {
          stopHls = await startHls(media, video.hls_url);
          if (abort.signal.aborted) return;
          void media.play().catch(() => undefined);
          const left = Math.max(GIVE_UP_MS - (Date.now() - started), 1000);
          if (await waitPlaying(media, left, abort.signal)) return;
        } catch {
          // no HLS either
        }
      }
      if (!abort.signal.aborted) setMode("none");
    })();
    return () => {
      abort.abort();
      whep?.close();
      stopHls?.();
      media.srcObject = null;
    };
  }, [video, serverDown, visible, attempt]);

  // Automatic new attempt every 10 s while there is no picture.
  useEffect(() => {
    if (mode !== "none" || !video) return;
    const timer = setTimeout(retry, RETRY_MS);
    return () => clearTimeout(timer);
  }, [mode, video, retry]);

  // Resolution and frame rate for the badge.
  useEffect(() => {
    if (mode !== "webrtc" && mode !== "hls") return;
    let frames = 0;
    let at = performance.now();
    const timer = setInterval(() => {
      const media = el.current;
      if (!media) return;
      const total = media.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
      const now = performance.now();
      const fps = Math.round(((total - frames) * 1000) / (now - at));
      frames = total;
      at = now;
      setInfo(media.videoWidth ? `${media.videoWidth}×${media.videoHeight} · ${fps} fps` : "");
    }, 1000);
    return () => clearInterval(timer);
  }, [mode]);

  const playing = mode === "webrtc" || mode === "hls";
  return (
    <div className="view-body video-box" data-testid="video-view" data-mode={mode}>
      <video
        ref={el}
        className="video-el"
        data-testid="video-el"
        muted
        playsInline
        autoPlay
        controls={false}
        disablePictureInPicture
        hidden={!playing && mode !== "connecting"}
      />
      {playing && (
        <span className="video-badge" data-testid="video-badge">
          {mode === "webrtc" ? t("views.video.live") : t("views.video.hls")}
          {info ? ` · ${info}` : ""}
        </span>
      )}
      {mode === "connecting" && <p className="video-note">{t("views.video.connecting")}</p>}
      {mode === "none" && (
        <div className="video-note view-empty">
          <p>{video ? t("views.video.failed") : t("views.video.none")}</p>
          <button type="button" className="btn" onClick={retry} data-testid="video-retry">
            {t("views.video.retry")}
          </button>
        </div>
      )}
    </div>
  );
}
