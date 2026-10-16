import { useTranslation } from "react-i18next";
import { useSessionStore } from "../../state/sessionStore";
import type { ViewProps } from "../registry";

/** Video arrives with the video ticket; the shell shows the frame and the empty state. */
export default function VideoView(_: ViewProps) {
  const { t } = useTranslation();
  const video = useSessionStore((s) => s.config?.video ?? null);
  return (
    <div className="view-body view-empty video-empty">
      <p>{video ? t("views.video.connecting") : t("views.video.none")}</p>
    </div>
  );
}
