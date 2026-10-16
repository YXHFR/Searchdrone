import {
  Activity,
  Gamepad2,
  type LucideIcon,
  Map as MapIcon,
  MessageSquare,
  UserSearch,
  Video,
} from "lucide-react";
import { type ComponentType, lazy } from "react";
import type { ViewId } from "../layout/layoutModel";
// The chat view is in the main bundle (it is the first thing an operator uses); the map,
// with MapLibre, and the other views load on demand.
import { ChatView } from "./chat/ChatView";

export interface ViewProps {
  vehicleId: string;
  isMaximized: boolean;
  isCompact: boolean;
}

export interface ViewDef {
  id: ViewId;
  titleKey: string;
  icon: LucideIcon;
  component: ComponentType<ViewProps>;
  minWidthPx: number;
}

export const VIEW_REGISTRY: Record<ViewId, ViewDef> = {
  chat: {
    id: "chat",
    titleKey: "views.chat.title",
    icon: MessageSquare,
    component: ChatView,
    minWidthPx: 300,
  },
  map: {
    id: "map",
    titleKey: "views.map.title",
    icon: MapIcon,
    component: lazy(() => import("./map/MapView")),
    minWidthPx: 280,
  },
  video: {
    id: "video",
    titleKey: "views.video.title",
    icon: Video,
    component: lazy(() => import("./video/VideoView")),
    minWidthPx: 280,
  },
  candidates: {
    id: "candidates",
    titleKey: "views.candidates.title",
    icon: UserSearch,
    component: lazy(() => import("./candidates/CandidatesView")),
    minWidthPx: 260,
  },
  status: {
    id: "status",
    titleKey: "views.status.title",
    icon: Activity,
    component: lazy(() => import("./status/StatusView")),
    minWidthPx: 280,
  },
  manual: {
    id: "manual",
    titleKey: "views.manual.title",
    icon: Gamepad2,
    component: lazy(() => import("./manual/ManualView")),
    minWidthPx: 300,
  },
};
