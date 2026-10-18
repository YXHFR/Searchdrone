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

export interface ViewDef {
  id: ViewId;
  titleKey: string;
  component: ComponentType<ViewProps>;
}

export const VIEW_REGISTRY: Record<ViewId, ViewDef> = {
  chat: { id: "chat", titleKey: "views.chat.title", component: ChatView },
  map: {
    id: "map",
    titleKey: "views.map.title",
    component: lazy(() => import("./map/MapView")),
  },
  video: {
    id: "video",
    titleKey: "views.video.title",
    component: lazy(() => import("./video/VideoView")),
  },
  candidates: {
    id: "candidates",
    titleKey: "views.candidates.title",
    component: lazy(() => import("./candidates/CandidatesView")),
  },
  status: {
    id: "status",
    titleKey: "views.status.title",
    component: lazy(() => import("./status/StatusView")),
  },
  manual: {
    id: "manual",
    titleKey: "views.manual.title",
    component: lazy(() => import("./manual/ManualView")),
  },
};
