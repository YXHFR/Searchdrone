// @vitest-environment node
// Every path the panel calls exists in the ground server's OpenAPI snapshot with that method.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { paths } from "./paths";

const SNAPSHOT = new URL(
  "../../../ground/tests/contract/snapshots/panel.openapi.json",
  import.meta.url,
);
const doc = JSON.parse(readFileSync(SNAPSHOT, "utf8")) as {
  paths: Record<string, Record<string, unknown>>;
};

function exists(method: string, url: string): boolean {
  const path = url.split("?")[0] ?? url;
  return Object.entries(doc.paths).some(
    ([p, ops]) =>
      method.toLowerCase() in ops && new RegExp(`^${p.replace(/\{[^}]+\}/g, "[^/]+")}$`).test(path),
  );
}

const ID = "0192f000-0000-7000-8000-000000000001";
const CALLS: [string, string][] = [
  ["GET", paths.health()],
  ["GET", paths.systemStatus()],
  ["GET", paths.panelConfig()],
  ["GET", paths.session()],
  ["POST", paths.session()],
  ["POST", paths.logout()],
  ["GET", paths.vehicleState("sim-01")],
  ...(["stop", "return_home", "resume", "takeover", "handback"] as const).map(
    (a): [string, string] => ["POST", paths.vehicleAction("sim-01", a)],
  ),
  ["GET", paths.vehicleActionOutcome("sim-01", ID)],
  ["POST", paths.allVehiclesAction("stop")],
  ["POST", paths.allVehiclesAction("return_home")],
  ["POST", paths.missionOpen("sim-01")],
  ["GET", paths.missionActive("sim-01")],
  ["GET", paths.mission(ID)],
  ["PATCH", paths.mission(ID)],
  ["PUT", paths.missionLimits(ID)],
  ["PUT", paths.missionAutonomy(ID)],
  ["POST", paths.missionClearance(ID)],
  ["POST", paths.missionEnd(ID)],
  ["GET", paths.missionCharters(ID)],
  ["GET", paths.missionSectors(ID)],
  ["GET", paths.missionApprovals(ID, "pending")],
  ["GET", paths.missionCommands(ID, "pending_approval")],
  ["POST", paths.missionCommands(ID)],
  ["POST", paths.approvalDecision(ID)],
  ["GET", paths.missionChatPage(ID, { limit: 50 })],
  ["POST", paths.missionChat(ID)],
  ["GET", paths.missionAnnotations(ID)],
  ["POST", paths.missionAnnotations(ID)],
  ["DELETE", paths.annotation(ID)],
  ["GET", paths.missionTrack(ID)],
];

describe("contract paths", () => {
  it.each(CALLS)("%s %s is in the OpenAPI snapshot", (method, url) => {
    expect(exists(method, url)).toBe(true);
  });
});
