// "Ta över" / "Lämna tillbaka": the take-over goes to its endpoint, the sticks go over the
// live socket with the take-over's id, and nothing more is sent after handing back.
import { observerSession, panelConfig, vehicleLive } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { showView } from "../helpers/safety";

const manualOps = (sent: Record<string, unknown>[]) => sent.filter((m) => m.op === "manual");

test("take over sends stick values with the take-over id; hand back stops them", async ({
  page,
  ground,
}) => {
  ground.setConfig({ features: { ...panelConfig.features, manual_control: true } });
  await ground.open();
  await ground.liveConnected();
  await showView(page, "manual");
  await page.getByTestId("takeover").click();
  await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/takeover").length).toBe(1);
  const takeoverId = "a1"; // the fake ground's first action id
  await expect.poll(() => manualOps(ground.wsSent).length, { timeout: 3_000 }).toBeGreaterThan(10);
  const ops = manualOps(ground.wsSent);
  const last = ops.at(-1) as Record<string, number | string>;
  expect(last.takeover_id).toBe(takeoverId);
  expect(last).toMatchObject({ roll: 0, pitch: 0, yaw: 0, throttle: 0.5, source: "touch" });
  const seqs = ops.map((o) => o.input_seq as number);
  expect([...seqs].sort((a, b) => a - b)).toEqual(seqs);
  expect(new Set(seqs).size).toBe(seqs.length);

  await page.getByTestId("handback").click();
  await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/handback").length).toBe(1);
  await page.waitForTimeout(200);
  const after = manualOps(ground.wsSent).length;
  await page.waitForTimeout(500);
  expect(manualOps(ground.wsSent).length).toBe(after);
  await expect(page.getByTestId("takeover")).toBeEnabled();
});

test("observers cannot take over", async ({ page, ground }) => {
  ground.setSession(observerSession);
  ground.setConfig({ features: { ...panelConfig.features, manual_control: true } });
  await ground.open();
  await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
  await showView(page, "manual");
  await expect(page.getByTestId("takeover")).toBeDisabled();
  await page.waitForTimeout(300);
  expect(manualOps(ground.wsSent)).toEqual([]);
});

test("STOP or landing ends the take-over on the ground: back to Ta över, no more sticks", async ({
  page,
  ground,
}) => {
  ground.setConfig({ features: { ...panelConfig.features, manual_control: true } });
  await ground.open();
  await ground.liveConnected();
  await showView(page, "manual");
  await page.getByTestId("takeover").click();
  const held = {
    active: true,
    takeover_id: "a1",
    by: "operator:demo",
    since: new Date().toISOString(),
    input_age_ms: 60,
  };
  ground.pushLive("vehicle", "vehicle.state", { ...vehicleLive, manual: held });
  await expect(page.getByTestId("handback")).toBeVisible();
  await expect.poll(() => manualOps(ground.wsSent).length).toBeGreaterThan(5);
  // Leaving the view (a phone tab) keeps the take-over.
  await showView(page, "chat");
  await showView(page, "manual");
  await expect(page.getByTestId("handback")).toBeVisible();

  // STOP ended it on the ground (no hand-back from this panel).
  await page.waitForTimeout(2_100); // past the grace after the take-over
  ground.pushLive("vehicle", "vehicle.state", {
    ...vehicleLive,
    manual: { active: false, takeover_id: null, by: null, since: null, input_age_ms: null },
  });
  await expect(page.getByTestId("takeover")).toBeVisible();
  await page.waitForTimeout(200);
  const after = manualOps(ground.wsSent).length;
  await page.waitForTimeout(500);
  expect(manualOps(ground.wsSent).length).toBe(after);
  expect(ground.posts("/api/v1/vehicles/sim-01/actions/handback")).toEqual([]);
});
