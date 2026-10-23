// The panel against the fake ground for B1-13: chat, approval cards, clearance, limits,
// autonomy, the mission card, order buttons and the AI notices.
import {
  approval,
  charter,
  chatMessage,
  command,
  observerSession,
  panelConfig,
  snapshot,
  square,
  telemetry,
  vehicleLive,
} from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { holdClick, showView } from "../helpers/safety";

const CHAT = "/api/v1/missions/M1/chat";
const UUIDV7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

test.describe("chat and approval cards", () => {
  test.beforeEach(async ({ page, ground }) => {
    await ground.open();
    await showView(page, "chat");
    await ground.liveConnected();
    await ground.resynced();
  });

  test("text + Enter sends one chat message, shown at once and once after the echo", async ({
    page,
    ground,
  }) => {
    const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
    await box.fill("lyft och sök av rutan norr om hem");
    await box.press("Enter");
    await expect(page.locator(".msg-operator")).toBeVisible();
    await expect.poll(() => ground.posts(CHAT).length).toBe(1);
    const body = ground.posts(CHAT)[0]?.body as {
      text: string;
      client_msg_id: string;
      annotation_ids: unknown[];
    };
    expect(body.text).toBe("lyft och sök av rutan norr om hem");
    expect(body.client_msg_id).toMatch(UUIDV7);
    expect(body.annotation_ids).toEqual([]);
    ground.pushLive(
      "chat",
      "chat.message",
      chatMessage({
        id: `c-${body.client_msg_id}`,
        role: "operator",
        text: body.text,
        refs: { client_msg_id: body.client_msg_id },
      }),
    );
    await expect(page.locator(".msg-operator")).toHaveCount(1);
  });

  test("the charter card shows the charter, approves with the chosen mode and never sends confirmations", async ({
    page,
    ground,
  }) => {
    ground.pushLive("mission", "charter.updated", charter());
    ground.pushLive("approvals", "approval.requested", approval());
    const card = page.getByTestId("approval-AP1");
    await expect(card).toContainText("Kräver ditt godkännande");
    await expect(card).toContainText("Ny ram");
    await expect(card).toContainText("ha, 100 × 100 m");
    await expect(card).toContainText("högst 8 m/s");
    await expect(card).toContainText("högst 500 m");
    await expect(card).toContainText("flyger hem vid 15 %");
    await expect(card).toContainText("gör klart steget och svävar");
    await expect(card).toContainText("Sök av sektor");
    // The owner chose approval directly on the card (no review step).
    await expect(card.getByRole("radio", { name: /B – AI:n flyger fritt/ })).toBeChecked();
    // A habitual tap approves nothing: the area has to be held for 0.8 s.
    await card.getByTestId("approve").click();
    await page.waitForTimeout(1_000);
    expect(ground.posts("/api/v1/approvals/AP1/decision")).toEqual([]);
    await holdClick(page, "approve", card);
    await expect.poll(() => ground.posts("/api/v1/approvals/AP1/decision").length).toBe(1);
    expect(ground.posts("/api/v1/approvals/AP1/decision")[0]?.body).toEqual({
      decision: "approve",
      subject_version: 1,
      autonomy_mode: "WITHIN_CHARTER",
    });
    expect(ground.calls(/\/(activate|approve)$/)).toEqual([]);
    await expect(card).toContainText("Godkänd av demo");
  });

  test("'ja' in the chat is never an approval", async ({ page, ground }) => {
    ground.pushLive("mission", "charter.updated", charter());
    ground.pushLive("approvals", "approval.requested", approval());
    await expect(page.getByTestId("approval-AP1")).toBeVisible();
    const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
    await box.fill("ja");
    await box.press("Enter");
    await expect.poll(() => ground.posts(CHAT).length).toBe(1);
    await page.waitForTimeout(300);
    expect(ground.calls(/^POST .*(\/decision|\/clearance|\/actions\/resume)$/)).toEqual([]);
  });

  test("limits from the server: above the cap cannot be approved; a looser change is marked", async ({
    page,
    ground,
  }) => {
    ground.pushLive("mission", "charter.updated", charter({}, { alt_max_agl_m: 150 }));
    ground.pushLive("approvals", "approval.requested", approval());
    const card = page.getByTestId("approval-AP1");
    await expect(card.getByTestId("card-problems")).toContainText("Kan inte godkännas");
    await expect(card.getByTestId("approve")).toBeDisabled();

    // An approved v1, then a proposed v2 that only changes on_ai_silent: the diff marks it looser.
    ground.pushLive(
      "mission",
      "charter.updated",
      charter({ status: "approved", install_status: "installed" }),
    );
    ground.pushLive("approvals", "approval.updated", approval({ status: "approved" }));
    ground.pushLive(
      "mission",
      "charter.updated",
      charter({ id: "C2", version: 2, approval_id: "AP3" }, { on_ai_silent: "return_home" }),
    );
    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({ id: "AP3", subject_id: "C2", subject_version: 2 }),
    );
    const change = page.getByTestId("approval-AP3");
    await expect(change).toContainText("Ändrad ram");
    await expect(change.getByTestId("charter-diff")).toContainText(
      "Om AI:n tystnar: continue_plan → return_home (lösare)",
    );
  });

  test("superseded and expired answers end the card; ALREADY_DECIDED reloads the cards", async ({
    page,
    ground,
  }) => {
    ground.pushLive("mission", "charter.updated", charter());
    ground.pushLive("approvals", "approval.requested", approval());
    ground.failNext("POST /api/v1/approvals/AP1/decision", 409, {
      code: "SUPERSEDED",
      detail_sv: "Ersatt",
    });
    const card = page.getByTestId("approval-AP1");
    await holdClick(page, "approve", card);
    await expect(card).toContainText("Ersatt av en nyare version");
    await expect(card.getByTestId("approve")).toHaveCount(0);

    ground.pushLive(
      "commands",
      "command.created",
      command({ skill: "goto", params: { lat: -35.36, lon: 149.165, alt_m: 40 } }),
    );
    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({ id: "AP2", kind: "command", subject_id: "K1" }),
    );
    ground.failNext("POST /api/v1/approvals/AP2/decision", 409, {
      code: "EXPIRED",
      detail_sv: "Gick ut",
    });
    const skill = page.getByTestId("approval-AP2");
    await expect(skill).toContainText("Order: Flyg till punkt");
    await skill.getByTestId("approve").click();
    await expect(skill).toContainText("Gick ut");

    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({ id: "AP4", kind: "command", subject_id: "K1" }),
    );
    ground.failNext("POST /api/v1/approvals/AP4/decision", 409, {
      code: "ALREADY_DECIDED",
      detail_sv: "Redan",
    });
    const before = ground.calls(/^GET .*\/approvals\?status=pending$/).length;
    await page.getByTestId("approval-AP4").getByTestId("approve").click();
    await expect
      .poll(() => ground.calls(/^GET .*\/approvals\?status=pending$/).length)
      .toBeGreaterThan(before);
  });

  test("an AI question is answered with in_reply_to, never with an approval", async ({
    page,
    ground,
  }) => {
    ground.pushLive(
      "chat",
      "chat.message",
      chatMessage({ id: "q1", kind: "question", text: "Ser du en röd jacka?" }),
    );
    const q = page.getByTestId("question");
    await expect(q).toContainText("Fråga till dig");
    await q.getByTestId("answer").click();
    await expect(page.getByTestId("reply-to")).toContainText("Ser du en röd jacka?");
    await page.keyboard.type("Nej");
    await page.keyboard.press("Enter");
    await expect.poll(() => ground.posts(CHAT).length).toBe(1);
    const reply = ground.posts(CHAT)[0]?.body as { in_reply_to?: string } | undefined;
    expect(reply?.in_reply_to).toBe("q1");
    expect(ground.calls(/\/approvals\//)).toEqual([]);
  });

  test("server text is shown as text, never as HTML", async ({ page, ground }) => {
    ground.pushLive(
      "chat",
      "chat.message",
      chatMessage({ id: "x1", text: '<img src=x onerror="window.__xss=1">' }),
    );
    await expect(page.getByText('<img src=x onerror="window.__xss=1">')).toBeVisible();
    expect(
      await page.evaluate(() => (window as unknown as { __xss?: number }).__xss),
    ).toBeUndefined();
    await expect(page.getByTestId("messages").locator("img")).toHaveCount(0);
  });

  test("AI status: thinking counter, command mode, the AI-down banner with the charter's behaviour", async ({
    page,
    ground,
  }) => {
    ground.pushLive("chat", "chat.brain_status", { state: "busy", typing: true });
    await expect(page.getByTestId("brain-row")).toContainText("AI:n tänker…");
    await page.waitForTimeout(2_100);
    await expect(page.getByTestId("brain-row")).toContainText(/[2-9] s/);
    ground.pushLive("chat", "chat.brain_status", { state: "ready", typing: false });
    await expect(page.getByTestId("brain-row")).toHaveText("");
    // The ground reports the AI process going away with chat.brain_status.
    ground.pushLive("chat", "chat.brain_status", { state: "down", typing: false });
    // Said once, in the command panel, which then holds the order buttons.
    await expect(page.getByTestId("ai-down-banner")).toContainText("AI:n svarar inte.");
    await expect(page.getByTestId("ai-down-banner").getByTestId("cmd-manual")).toBeVisible();
    await expect(page.getByTestId("brain-row")).toHaveText("");
    await expect(page.getByTestId("pill-ai")).toContainText("AI borta");
    ground.pushLive("chat", "chat.brain_status", { state: "ready", typing: false });
    await expect(page.getByTestId("ai-down-banner")).toHaveCount(0);

    ground.mission = { ...ground.mission, active_charter_id: "C1" };
    ground.pushLive(
      "mission",
      "charter.updated",
      charter({ status: "approved", install_status: "installed" }),
    );
    ground.pushLive("mission", "mission.updated", ground.mission);
    ground.pushLive("alerts", "alert", {
      id: null,
      severity: "critical",
      code: "BRAIN_DISCONNECTED",
      message_sv: "AI borta",
      data: {},
    });
    const banner = page.getByTestId("ai-down-banner");
    await expect(banner).toContainText("AI:n svarar inte.");
    await expect(banner.getByTestId("cmd-behaviour")).toHaveText(
      "Drönaren gör klart steget och svävar sedan.",
    );
    await expect(banner).toContainText("STOPP och FLYG HEM fungerar.");
    await expect(banner).not.toContainText(/\d+:\d\d/);
    ground.pushLive("system", "system.status", {
      version: "0.0.0",
      profile: "sim",
      internet: "unknown",
      operator_present: true,
      parts: { brain: { state: "ok", since: "2026-10-16T10:00:00Z" } },
    });
    await expect(banner).toHaveCount(0);
    await expect(page.getByText(/AI får styra|moln-AI/i)).toHaveCount(0);
  });

  test("'Ändra' puts 'Ändra ramen: ' in the composer and sends nothing", async ({
    page,
    ground,
  }) => {
    ground.pushLive("mission", "charter.updated", charter());
    ground.pushLive("approvals", "approval.requested", approval());
    // The probability map's images load in the background and are not sent by this click.
    const sent = () => ground.requests.filter((r) => !r.path.includes("/probability-map.png"));
    const before = sent().length;
    await page.getByTestId("approval-AP1").getByTestId("edit-charter").click();
    const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
    await expect(box).toHaveValue("Ändra ramen: ");
    await expect(box).toBeFocused();
    expect(sent().length).toBe(before);
  });

  test("a proposal to resume draws the Resume button; typing 'fortsätt' never resumes", async ({
    page,
    ground,
  }) => {
    ground.pushLive("vehicle", "vehicle.state", {
      ...vehicleLive,
      airborne: true,
      snapshot: snapshot({
        executor_state: "HELD",
        held_by: "operator_stop",
        resume_available: true,
      }),
    });
    const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
    await box.fill("fortsätt");
    await box.press("Enter");
    await expect.poll(() => ground.posts(CHAT).length).toBe(1);
    ground.pushLive(
      "chat",
      "chat.message",
      chatMessage({
        id: "p1",
        kind: "proposal",
        text: "Vill du fortsätta?",
        refs: { proposed_action: "resume" },
      }),
    );
    const msg = page.locator(".msg", { hasText: "Vill du fortsätta?" });
    await expect(msg.getByTestId("resume")).toBeVisible();
    expect(ground.posts("/api/v1/vehicles/sim-01/actions/resume")).toEqual([]);
    await page.waitForTimeout(2_100);
    await msg.getByTestId("resume").click();
    await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/resume").length).toBe(1);
  });
});

test.describe("pending approvals while the chat is hidden", () => {
  test("badge and toast; 'Visa' brings up the chat and focuses the card", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await showView(page, "map");
    await ground.liveConnected();
    await ground.resynced();
    ground.pushLive("commands", "command.created", command());
    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({ id: "AP2", kind: "command", subject_id: "K1" }),
    );
    await expect(page.getByTestId("pending-badge")).toContainText("1", { timeout: 1_000 });
    await expect(page.getByTestId("toasts")).toContainText(
      "Väntar på ditt godkännande: Order: Starta",
    );
    await page.getByTestId("toast-show").click();
    await expect(page.getByTestId("approval-AP2")).toBeFocused();
  });
});

test.describe("clearance, limits, autonomy, mission card and order buttons", () => {
  const onGround = {
    ...vehicleLive,
    airborne: false,
    snapshot: snapshot({ executor_state: "PREFLIGHT" }),
  };

  async function readyMission(ground: import("../helpers/fakeGround").FakeGround) {
    ground.mission = {
      ...ground.mission,
      status: "READY",
      active_charter_id: "C1",
      autonomy_mode: "WITHIN_CHARTER",
    };
    ground.charters = [charter({ status: "approved", install_status: "installed" })];
    ground.setConfig({
      features: { ...panelConfig.features, order_buttons: true, manual_control: true },
    });
    await ground.open();
    await ground.liveConnected();
    await ground.resynced();
    ground.pushLive("vehicle", "vehicle.state", onGround);
    ground.pushLive(
      "telemetry",
      "telemetry",
      telemetry({ armed: false, landed_state: "on_ground", alt_rel_home_m: 0 }),
    );
  }

  test("order buttons: Starta on the ground; pins, the next sector and Sväva här in the air", async ({
    page,
    ground,
  }) => {
    ground.sectors = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: square(50, 0, 200),
          properties: { sector_id: "A", name: "A", status: "pending", progress: 0 },
        },
      ],
    };
    await readyMission(ground);
    await showView(page, "chat");
    const orders = page.locator(".orders");
    await expect(orders.getByTestId("order-takeoff")).toBeVisible();
    await expect(orders.getByTestId("order-hold")).toHaveCount(0);
    ground.pushLive("vehicle", "vehicle.state", {
      ...onGround,
      airborne: true,
      snapshot: snapshot({ executor_state: "HOLD" }),
    });
    ground.pushLive("telemetry", "telemetry", telemetry());
    await expect(orders.getByTestId("order-takeoff")).toHaveCount(0);
    await expect(orders.getByTestId("order-survey_sector")).toHaveText("Sök av A");
    await orders.getByTestId("order-hold").click();
    await expect
      .poll(() => ground.posts("/api/v1/missions/M1/commands").map((r) => r.body))
      .toEqual([expect.objectContaining({ skill: "hold", params: {} })]);
  });

  test("'Fritt runt drönaren': box first, one POST with {}, countdown, reset when consumed", async ({
    page,
    ground,
  }) => {
    await readyMission(ground);
    await showView(page, "chat");
    const cc = page.getByTestId("clearance");
    await expect(cc).toBeVisible();
    await expect(cc.getByTestId("clearance-button")).toBeDisabled();
    await cc.getByRole("checkbox").check();
    await cc.getByTestId("clearance-button").click();
    await expect.poll(() => ground.posts("/api/v1/missions/M1/clearance").length).toBe(1);
    expect(ground.posts("/api/v1/missions/M1/clearance")[0]?.body).toEqual({});
    expect(ground.calls(/\/approvals\//)).toEqual([]);
    ground.pushLive("mission", "clearance.updated", {
      status: "active",
      confirmed_by: "operator:demo",
      valid_until: new Date(Date.now() + 120_000).toISOString(),
    });
    await expect(page.getByTestId("clearance")).toContainText(/Klartecken gäller (1:59|2:00)/);
    ground.pushLive("mission", "clearance.updated", {
      status: "consumed",
      confirmed_by: null,
      valid_until: null,
    });
    await expect(page.getByTestId("clearance").getByRole("checkbox")).not.toBeChecked();
    // Not offered once the vehicle is armed.
    ground.pushLive("telemetry", "telemetry", telemetry({ armed: true }));
    await expect(page.getByTestId("clearance")).toHaveCount(0);
  });

  test("mode A: a take-off card needs the clearance first, then approves without confirmations", async ({
    page,
    ground,
  }) => {
    await readyMission(ground);
    await showView(page, "chat");
    ground.pushLive("commands", "command.created", command());
    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({
        id: "AP2",
        kind: "command",
        subject_id: "K1",
        required_confirmations: ["clear_around_drone"],
      }),
    );
    const card = page.getByTestId("approval-AP2");
    await expect(card.getByTestId("clearance")).toBeVisible();
    await expect(card.getByTestId("approve")).toBeDisabled();
    await card.getByRole("checkbox").check();
    await card.getByTestId("clearance-button").click();
    await expect.poll(() => ground.posts("/api/v1/missions/M1/clearance").length).toBe(1);
    await expect(card.getByTestId("approve")).toBeEnabled();
    await card.getByTestId("approve").click();
    await expect.poll(() => ground.posts("/api/v1/approvals/AP2/decision").length).toBe(1);
    expect(ground.posts("/api/v1/approvals/AP2/decision")[0]?.body).toEqual({
      decision: "approve",
      subject_version: 1,
    });
  });

  test("limits form: prefilled, above the cap blocks Save, one PUT with all limits; server errors shown", async ({
    page,
    ground,
  }) => {
    await readyMission(ground);
    await showView(page, "status");
    await page.getByTestId("limits-open").click();
    const form = page.getByTestId("limits-form");
    await expect(form.getByTestId("limit-max_alt_agl_m")).toHaveValue("120");
    await form.getByTestId("limit-max_alt_agl_m").fill("130");
    await expect(form.getByTestId("limits-save")).toBeDisabled();
    await expect(form).toContainText("Över det absoluta taket (120)");
    await form.getByTestId("limit-max_alt_agl_m").fill("100");
    ground.failNext("PUT /api/v1/missions/M1/limits", 409, {
      code: "LIMITS_BELOW_ACTIVE_CHARTER",
      detail_sv: "Ramen når 60 m.",
    });
    await form.getByTestId("limits-save").click();
    await expect(form).toContainText("Den godkända ramen ligger utanför de nya gränserna.");
    await form.getByTestId("limits-save").click();
    await expect(form).toHaveCount(0);
    const puts = ground.calls(/^PUT .*\/limits$/);
    expect(puts).toHaveLength(2);
    expect(puts[1]?.body).toEqual({
      max_alt_agl_m: 100,
      max_speed_ms: 15,
      max_range_from_pilot_m: 500,
      battery_rtl_pct: 15,
    });
  });

  test("autonomy changes only through the dialog; the mission card ends with its own dialog", async ({
    page,
    ground,
  }) => {
    await readyMission(ground);
    await showView(page, "status");
    let dialogs = 0;
    page.on("dialog", () => {
      dialogs += 1;
    });
    await page.getByTestId("autonomy-change").click();
    await page.getByTestId("autonomy-dialog").getByRole("button", { name: "Byt" }).click();
    await expect.poll(() => ground.calls(/^PUT .*\/autonomy$/).length).toBe(1);
    expect(ground.calls(/^PUT .*\/autonomy$/)[0]?.body).toEqual({ mode: "APPROVE_EACH" });
    await expect(page.getByTestId("mission-card")).not.toContainText("Nytt uppdrag");
    ground.failNext("POST /api/v1/missions/M1/end", 409, {
      code: "VEHICLE_AIRBORNE",
      detail_sv: "I luften",
    });
    await page.getByTestId("mission-end").click();
    await page.getByTestId("end-confirm").click();
    await expect(page.getByTestId("mission-card")).toContainText(
      "Drönaren är i luften – tryck FLYG HEM först.",
    );
    expect(dialogs).toBe(0);
  });

  test("order buttons post the pilot's own order through the gate", async ({ page, ground }) => {
    await readyMission(ground);
    await showView(page, "chat");
    await page.getByTestId("order-takeoff").click();
    await expect.poll(() => ground.posts("/api/v1/missions/M1/commands").length).toBe(1);
    const b = ground.posts("/api/v1/missions/M1/commands")[0]?.body as {
      skill: string;
      params: unknown;
      client_command_id: string;
    };
    expect(b.skill).toBe("takeoff");
    expect(b.params).toEqual({ alt_m: 40 });
    expect(b.client_command_id).toMatch(UUIDV7);
  });

  test("observers get no clearance, limits, drawing or order buttons and never open a mission", async ({
    page,
    ground,
  }) => {
    ground.setSession(observerSession);
    await readyMission(ground);
    await showView(page, "chat");
    await expect(page.getByTestId("clearance")).toHaveCount(0);
    await expect(page.getByTestId("order-takeoff")).toHaveCount(0);
    await showView(page, "status");
    await expect(page.getByTestId("limits-open")).toHaveCount(0);
    expect(ground.calls(/missions\/open/)).toEqual([]);
    expect(
      ground.calls(/^GET \/api\/v1\/missions\/active\?vehicle_id=sim-01$/).length,
    ).toBeGreaterThan(0);
  });
});
