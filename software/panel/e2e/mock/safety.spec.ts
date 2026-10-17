import { observerSession, outcome, vehicleLive } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, isMobile } from "../helpers/safety";

const UUIDV7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const VIEWS = ["chat", "map", "video", "candidates", "status", "manual"] as const;
const STOP = "/api/v1/vehicles/sim-01/actions/stop";
const HOME = "/api/v1/vehicles/sim-01/actions/return_home";

test.describe("safety bar is always visible", () => {
  test("in every view, in full screen and on phones in every tab", async ({ page, ground }) => {
    await ground.open();
    await expectSafetyVisible(page, "start");
    if (isMobile(page)) {
      for (const v of VIEWS) {
        await page.getByTestId(`tab-${v}`).click();
        await expectSafetyVisible(page, `tab ${v}`);
      }
      await page.getByTestId("tab-chat").click();
      await page.getByTestId("pip-toggle").click();
      await expectSafetyVisible(page, "pip");
      // The tab row and the safety bar keep a gap with a divider, so a missed tab tap never
      // hits a safety button.
      const tabs = await page.getByTestId("mobile-tabs").boundingBox();
      const bar = await page.getByTestId("safety-bar").boundingBox();
      expect((bar?.y ?? 0) - ((tabs?.y ?? 0) + (tabs?.height ?? 0))).toBeGreaterThanOrEqual(12);
      await expect(page.locator(".tabs-gap")).toBeVisible();
      return;
    }
    for (const v of VIEWS) {
      const first = page.locator("[data-view]").first();
      if ((await first.getAttribute("data-view")) !== v)
        await first.locator("select").first().selectOption(v);
      await page.getByTestId(`maximize-${v}`).click();
      await expectSafetyVisible(page, `full screen ${v}`);
      await page.keyboard.press("Escape");
    }
    await page.getByTestId("layout-menu").click();
    await expectSafetyVisible(page, "layout menu open");
  });

  test("while the panel is still loading", async ({ page, ground }) => {
    ground.hangNext("GET /api/v1/session", 1);
    await ground.open();
    await expect(page.getByText("Laddar…")).toBeVisible();
    await expectSafetyVisible(page, "loading");
  });

  test("with the chat field focused", async ({ page, ground }) => {
    await ground.open();
    if (isMobile(page)) await page.getByTestId("tab-chat").click();
    await page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" }).focus();
    await expectSafetyVisible(page, "chat focus");
  });

  test("on the login page, with the warning text", async ({ page, ground }) => {
    ground.setSession(null);
    await ground.open();
    await expect(page.getByRole("heading", { name: "Logga in" })).toBeVisible();
    await expect(
      page.getByText(
        "Inte inloggad – STOPP och FLYG HEM fungerar inte förrän du loggat in. Använd RC.",
      ),
    ).toBeVisible();
    await expectSafetyVisible(page, "login");
  });

  test("on the re-login sheet, and STOP still sends", async ({ page, ground }) => {
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    // The session runs out in flight: the next API call (here a chat message) gets 401.
    ground.setSession(null);
    if (isMobile(page)) await page.getByTestId("tab-chat").click();
    ground.failNext("POST /api/v1/missions/M1/chat", 401, { code: "UNAUTHENTICATED" });
    const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
    await box.fill("hej");
    await box.press("Enter");
    await expect(page.getByTestId("reauth-sheet")).toBeVisible();
    await expect(page.getByTestId("topbar")).toBeVisible();
    await expectSafetyVisible(page, "reauth");
    await page.getByTestId("safety-stop").click();
    await expect.poll(() => ground.posts(STOP).length).toBe(1);
  });

  test("on the emergency page, STOP uses stop_all when no vehicle is known", async ({
    page,
    ground,
  }) => {
    ground.respond("GET /api/v1/panel/config", () => ({ status: 500, body: {} }), 100);
    await ground.open();
    await expect(page.getByTestId("emergency")).toBeVisible({ timeout: 3_500 });
    await expectSafetyVisible(page, "emergency");
    await page.getByTestId("safety-stop").click();
    await expect.poll(() => ground.posts("/api/v1/actions/stop_all").length).toBe(1);
  });

  test("STOP is visible and enabled within 1.5 s of loading", async ({ page, ground }) => {
    test.skip(page.viewportSize()?.width !== 1280, "desktop only");
    await ground.open();
    await expect(page.getByTestId("safety-stop")).toBeEnabled();
    // Time since navigation start, measured in the page (not the test runner's clock).
    const ms = await page.evaluate(() => performance.now());
    expect(ms).toBeLessThanOrEqual(1_500);
  });

  test("tab order: skip link, then STOP, then RETURN HOME", async ({ page, ground }) => {
    test.skip(isMobile(page), "keyboard on desktop");
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    await page.keyboard.press("Tab");
    await expect(page.getByText("Hoppa till innehållet")).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(page.getByTestId("safety-stop")).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(page.getByTestId("safety-home")).toBeFocused();
  });
});

test.describe("STOP and RETURN HOME go straight to their endpoints", () => {
  for (const [name, testId, path, action] of [
    ["STOP", "safety-stop", STOP, "stop"],
    ["RETURN HOME", "safety-home", HOME, "return_home"],
  ] as const) {
    test(`${name}: one press = one POST, never via chat, honest acknowledgement`, async ({
      page,
      ground,
    }) => {
      await ground.open();
      await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
      const before = ground.requests.length;
      await page.getByTestId(testId).click();
      await expect.poll(() => ground.posts(path).length).toBe(1);
      const post = ground.posts(path)[0];
      const body = (post?.body ?? {}) as { client_action_id?: string };
      expect(Object.keys(body)).toEqual(["client_action_id"]);
      expect(body.client_action_id).toMatch(UUIDV7);
      expect(post?.headers["x-sd-client"]).toBe("panel");
      await expect(page.getByTestId(testId)).toContainText("Skickat – väntar på drönaren");

      // Onboard receipt is not an acknowledgement.
      ground.pushLive(
        "vehicle",
        "vehicle.action",
        outcome("a1", { action, paths: { mavlink: "queued", sdlink: "sent" } }),
      );
      await expect(page.getByTestId(testId)).toContainText("Ombord har tagit emot");
      await expect(page.getByTestId(testId)).not.toContainText("Kvitterat");

      ground.pushLive(
        "vehicle",
        "vehicle.action",
        outcome("a1", { action, result: "acked", mode_confirmed: "BRAKE" }),
      );
      await expect(page.getByTestId(testId)).toContainText("Kvitterat", { timeout: 500 });

      // Nothing went to the chat, approvals, commands or clearance.
      await page.waitForTimeout(500);
      const after = ground.requests.slice(before);
      expect(after.filter((r) => /\/chat|\/approvals|\/commands|\/clearance/.test(r.path))).toEqual(
        [],
      );
      expect(after.filter((r) => r.method !== "GET").map((r) => r.path)).toEqual([path]);

      // A second press is a new press with a new id.
      await page.getByTestId(testId).click();
      await expect.poll(() => ground.posts(path).length).toBe(2);
      const ids = ground
        .posts(path)
        .map((r) => (r.body as { client_action_id: string }).client_action_id);
      expect(ids[0]).not.toBe(ids[1]);
    });
  }

  test("no acknowledgement → yellow warning, never 'Kvitterat'", async ({ page, ground }) => {
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    await page.getByTestId("safety-stop").click();
    await expect(page.getByTestId("safety-stop")).toContainText("kvittens saknas", {
      timeout: 4_000,
    });
    await page.waitForTimeout(2_000);
    await expect(page.getByTestId("safety-stop")).not.toContainText("Kvitterat");
    await page.getByRole("button", { name: "Stäng meddelandet" }).click();
    await expect(page.getByTestId("safety-stop")).toContainText("Stannar och svävar");
  });

  test("WebSocket down: banner, and the acknowledgement is polled over HTTP", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    ground.refuseSockets = true;
    await ground.dropSocket();
    await expect(page.getByTestId("connection-banner")).toBeVisible({ timeout: 2_500 });
    ground.respond(
      "GET /api/v1/vehicles/sim-01/actions/a1",
      () => ({ status: 200, body: outcome("a1") }),
      2,
    );
    ground.respond(
      "GET /api/v1/vehicles/sim-01/actions/a1",
      () => ({ status: 200, body: outcome("a1", { result: "acked" }) }),
      10,
    );
    await page.getByTestId("safety-stop").click();
    await expect(page.getByTestId("safety-stop")).toContainText("Kvitterat", { timeout: 3_000 });
    expect(
      ground.calls(/^GET \/api\/v1\/vehicles\/sim-01\/actions\/a1$/).length,
    ).toBeGreaterThanOrEqual(2);
  });

  test("no path to the vehicle (503) → clear message, exactly one POST", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    ground.failNext(`POST ${STOP}`, 503, { code: "NO_PATH", detail_sv: "Ingen väg" });
    await page.getByTestId("safety-stop").click();
    await expect(page.getByTestId("safety-stop")).toContainText("Ingen förbindelse med drönaren", {
      timeout: 500,
    });
    await page.waitForTimeout(1_000);
    expect(ground.posts(STOP)).toHaveLength(1);
  });

  test("timeouts are retried with the same id", async ({ page, ground }) => {
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    ground.hangNext(`POST ${STOP}`, 2);
    await page.getByTestId("safety-stop").click();
    await expect(page.getByTestId("safety-stop")).toContainText("Inget svar – försöker igen");
    await expect(page.getByTestId("safety-stop")).toContainText("Skickat", { timeout: 9_000 });
    const ids = ground
      .posts(STOP)
      .map((r) => (r.body as { client_action_id: string }).client_action_id);
    expect(ids).toHaveLength(3);
    expect(new Set(ids).size).toBe(1);
  });

  test("observers may press STOP but not RETURN HOME", async ({ page, ground }) => {
    ground.setSession(observerSession);
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    await expect(page.getByTestId("safety-home")).toBeDisabled();
    await expect(page.getByTestId("safety-home")).toContainText("Bara piloten kan flyga hem");
    await page.getByTestId("safety-stop").click();
    await expect.poll(() => ground.posts(STOP).length).toBe(1);
    // Observers load the mission read-only and never open one.
    expect(ground.calls(/^POST .*\/missions\/open$/)).toEqual([]);
    expect(ground.calls(/^GET \/api\/v1\/missions\/active/).length).toBeGreaterThanOrEqual(1);
  });

  test("no keyboard shortcut triggers STOP", async ({ page, ground }) => {
    test.skip(isMobile(page), "keyboard on desktop");
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    await page.locator("body").press("Space");
    await page.locator("body").press("Shift+H");
    await page.waitForTimeout(300);
    expect(ground.calls(/actions\/(stop|return_home)/)).toEqual([]);
  });
});

test.describe("Resume and manual control", () => {
  const held = (held_by: string) => ({
    ...vehicleLive,
    airborne: true,
    snapshot: { executor_state: "HELD", held_by, resume_available: true },
  });

  test("Resume appears after a hold, is armed after 2 s and posts only to /actions/resume", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
    if (isMobile(page)) await page.getByTestId("tab-chat").click();
    await ground.liveConnected();
    ground.pushLive("vehicle", "vehicle.state", held("operator_stop"));
    const resume = page.getByTestId("view-chat").getByTestId("resume");
    await expect(resume).toBeVisible();
    await expect(resume).toBeDisabled();
    await expect(resume).toBeEnabled({ timeout: 3_000 });
    await resume.click();
    await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/resume").length).toBe(1);
    expect(ground.calls(/^POST .*\/chat$/)).toEqual([]);
    // Resume is never in the safety bar.
    await expect(page.getByTestId("safety-bar").getByTestId("resume")).toHaveCount(0);
  });

  test("Resume also after other interruptions, never during a return home", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
    if (isMobile(page)) await page.getByTestId("tab-chat").click();
    await ground.liveConnected();
    ground.pushLive("vehicle", "vehicle.state", held("onboard_restart"));
    await expect(page.getByTestId("resume")).toBeVisible();
    ground.pushLive("vehicle", "vehicle.state", held("operator_home"));
    await expect(page.getByTestId("resume")).toHaveCount(0);
    await expect(page.getByText("Tryck STOPP först för att avbryta hemflygningen.")).toBeVisible();
  });

  test("observers see no Resume and cannot take over", async ({ page, ground }) => {
    ground.setSession(observerSession);
    await ground.open();
    await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
    if (isMobile(page)) await page.getByTestId("tab-chat").click();
    await ground.liveConnected();
    ground.pushLive("vehicle", "vehicle.state", held("operator_stop"));
    await expect(page.getByTestId("resume-row")).toBeVisible();
    await expect(page.getByTestId("resume")).toHaveCount(0);
    if (isMobile(page)) await page.getByTestId("tab-manual").click();
    else await page.locator("[data-view]").first().locator("select").first().selectOption("manual");
    await expect(page.getByTestId("takeover")).toBeDisabled();
    await expect(page.getByText("Bara piloten kan styra för hand.")).toBeVisible();
  });

  test("manual control and order buttons are placeholders that send nothing", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
    if (isMobile(page)) await page.getByTestId("tab-chat").click();
    for (const b of await page.getByTestId("order-button").all()) await expect(b).toBeDisabled();
    if (isMobile(page)) await page.getByTestId("tab-manual").click();
    else await page.locator("[data-view]").first().locator("select").first().selectOption("manual");
    await expect(page.getByTestId("takeover")).toBeDisabled();
    await expect(page.getByTestId("manual-axes")).toContainText("Gas 0.00");
    const before = ground.requests.filter((r) => r.method !== "GET").length;
    await page.mouse.click(10, 300);
    expect(ground.requests.filter((r) => r.method !== "GET").length).toBe(before);
  });
});
