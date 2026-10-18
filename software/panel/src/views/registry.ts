import { type ComponentType, lazy } from "react";
// The chat view is in the main bundle (it is the first thing an operator uses); the map,
// with MapLibre, and the other views load on demand.
import { ChatView } from "./chat/ChatView";

export type ViewId = "chat" | "map" | "video" | "candidates" | "status" | "manual";

export interface ViewProps {
  vehicleId: string;
  /** Drawn small (picture-in-picture or a narrow column): fewer extras. */
  isCompact: boolean;
}

export const VIEW_REGISTRY: Record<ViewId, ComponentType<ViewProps>> = {
  chat: ChatView,
  map: lazy(() => import("./map/MapView")),
  video: lazy(() => import("./video/VideoView")),
  candidates: lazy(() => import("./candidates/CandidatesView")),
  status: lazy(() => import("./status/StatusView")),
  manual: lazy(() => import("./manual/ManualView")),
};
