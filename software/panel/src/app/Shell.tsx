import { useEffect } from "react";
import { startLive } from "../api/live";
import { ReauthSheet } from "../auth/ReauthSheet";
import { AiDownBanner } from "../header/AiDownBanner";
import { Toasts } from "../header/Toasts";
import { ConnectionBanner, TopBar } from "../header/TopBar";
import { LayoutRoot } from "../layout/LayoutRoot";
import { viewportFor } from "../layout/layoutModel";
import { MobileLayout } from "../layout/MobileLayout";
import { useLayoutStore } from "../state/layoutStore";
import { useSessionStore } from "../state/sessionStore";

function useViewportClass(): void {
  useEffect(() => {
    const update = (): void =>
      useLayoutStore.getState().setViewport(viewportFor(window.innerWidth));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
}

export function Shell({ vehicleId }: { vehicleId: string }) {
  useViewportClass();
  const vp = useLayoutStore((s) => s.vp);
  const needsReauth = useSessionStore((s) => s.needsReauth);
  useEffect(() => startLive(vehicleId), [vehicleId]);
  return (
    <div className={`shell shell-${vp}`}>
      <TopBar />
      <ConnectionBanner />
      <AiDownBanner />
      {vp === "mobile" ? (
        <MobileLayout vehicleId={vehicleId} />
      ) : (
        <LayoutRoot vehicleId={vehicleId} />
      )}
      <Toasts />
      {needsReauth ? <ReauthSheet /> : null}
    </div>
  );
}
