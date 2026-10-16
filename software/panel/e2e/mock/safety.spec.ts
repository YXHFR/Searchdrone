import { observerSession, outcome } from "../fixtures";
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
      // The tab row and STOP keep a gap so a missed tab tap never hits a safety button.
      const tabs = await page.getByTestId("mobile-tabs").boundingBox();
      const stop = await page.getByTestId("safety-stop").boundingBox();
      expect((stop?.y ?? 0) - ((tabs?.y ?? 0) + (tabs?.height ?? 0))).toBeGreaterThanOrEqual(12);
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

  test("with large desktop and phone sizes", async ({ page, ground }) => {
    test.skip(
      page.viewportSize()?.width !== 1280 && page.viewportSize()?.width !== 360,
      "runs in desktop and phone-360",
    );
    for (const size of [
      { width: 1920, height: 1080 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(size);
      await ground.open();
      await expectSafetyVisible(page, `${size.width}x${size.height}`);
    }
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
    const t0 = Date.now();
    await ground.open();
    await expect(page.getByTestId("safety-stop")).toBeEnabled();
    expect(Date.now() - t0).toBeLessThanOrEqual(1_500);
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
