import { useTranslation } from "react-i18next";
import { ViewSlot } from "../app/ViewSlot";
import { AiStatus } from "../header/TopBar";
import { FindsPanel, MissionPanel } from "../mission/MissionPanel";
import { type SideTab, useShellStore } from "../state/shellStore";
import { useUiStore } from "../state/uiStore";
import { CommandPanel, useAiGone } from "./CommandPanel";
import { EventsList } from "./EventsList";

/** The conversation column (desktop and medium) and the phone's sheet share these tabs. */
export function SideTabs({ withMission }: { withMission: boolean }) {
  const { t } = useTranslation();
  const tab = useShellStore((s) => s.tab);
  const events = useUiStore((s) => s.eventLog.length);
  const tabs: SideTab[] = withMission ? ["chat", "events", "mission"] : ["chat", "events"];
  return (
    <div className="side-head">
      <div className="side-tabs" role="tablist" aria-label={t("side.label")}>
        {tabs.map((id) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`side-tab-${id}`}
            aria-selected={tab === id}
            aria-controls="side-body"
            data-testid={`side-tab-${id}`}
            onClick={() => useShellStore.getState().showTab(id)}
          >
            {t(`side.${id}`)}
            {id === "events" && events > 0 ? <span className="count num">{events}</span> : null}
          </button>
        ))}
      </div>
      <AiStatus />
    </div>
  );
}

/** The selected tab's content. The chat stays mounted so a draft and the scroll survive. */
export function SideBody({ vehicleId, withMission }: { vehicleId: string; withMission: boolean }) {
  const tab = useShellStore((s) => s.tab);
  const gone = useAiGone();
  const shown = !withMission && tab === "mission" ? "chat" : tab;
  return (
    <div className="side-body" id="side-body" role="tabpanel" aria-labelledby={`side-tab-${shown}`}>
      <div className="side-chat" data-testid="view-chat" hidden={shown !== "chat"}>
        {gone ? <CommandPanel /> : null}
        <ViewSlot view="chat" vehicleId={vehicleId} isCompact={false} />
      </div>
      {shown === "events" ? <EventsList /> : null}
      {shown === "mission" ? (
        <div className="side-mission">
          <MissionPanel />
          <FindsPanel />
        </div>
      ) : null}
    </div>
  );
}
