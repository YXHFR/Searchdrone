// Titta närmare: the pilot sends the drone for a closer look at a candidate. The button is the
// pilot's and only while the drone flies with a link and no other look is open; the question
// comes first and says what the drone will do in this operation mode, and a gate's no stays in
// it. During the look the strip shows the step, the height and the floor; the drone's "Gå
// lägre?" is answered by the pilot only; when the steps are done a countdown runs to the
// decision, with Fortsätt söka nu and Ta fler bilder. STOP and FLY HOME are on screen all the
// way. The top bar names the operation mode; Inställningar holds the live overflight permit.
import type { Page } from "@playwright/test";
import type {
  CandidateCard,
  CandidateDetail,
  Inspection,
  MissionCharter,
} from "@searchdrone/schemas";
import { approval, charter, command, observerSession, panelConfig, vehicleLive } from "../fixtures";
import { expect, type FakeGround, test } from "../helpers/fakeGround";
import { expectSafetyVisible, showView } from "../helpers/safety";

const AT = { lat: 59.28842, lon: 18.15953 };
const inSeconds = (s: number): string => new Date(Date.now() + s * 1_000).toISOString();

function card(seq: number, over: Partial<CandidateCard> = {}): CandidateCard {
  return {
    candidate_id: `0192f4a1-7b90-7a22-b1c3-0000000000${String(seq).padStart(2, "0")}`,
    seq,
    rev: 1,
    state: "ready",
    priority: 0.8,
    priority_class: "high",
    best_detector_score: 0.7,
    first_seen_utc: new Date(Date.now() - 4 * 60_000).toISOString(),
    last_seen_utc: new Date().toISOString(),
    ...AT,
    n_seen: 2,
    n_in_view: 3,
    sigma95_m: 6,
    thumb_detection_id: `d-${seq}`,
    engine: "strong",
    person: "yes",
    ...over,
  };
}

function extra(seq: number): Pick<CandidateDetail, "observations" | "assessments"> {
  return {
    observations: [
      {
        detection_id: `d-${seq}`,
        captured_utc: new Date(Date.now() - 60_000).toISOString(),
        distance_m: 1,
        has_crop: true,
        label: "person",
        ...AT,
        px_size: 40,
        score: 0.7,
        sigma95_m: 6,
      },
    ],
    assessments: [],
  };
}

/** The drone flies under an approved charter in `body`'s mode, K3 waits in the queue. */
async function flying(
  ground: FakeGround,
  body: Partial<MissionCharter> = { operation_mode: "exercise", overflight_allowed: true },
  cards: CandidateCard[] = [card(3)],
): Promise<void> {
  ground.setConfig({ features: { ...panelConfig.features, candidates: true } });
  ground.charters = [charter({ status: "approved", install_status: "installed" }, body)];
  ground.mission = { ...ground.mission, status: "ACTIVE", active_charter_id: "C1" };
  ground.candidates = cards;
  for (const c of cards) ground.candidateDetail[c.candidate_id] = extra(c.seq);
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  ground.pushLive("vehicle", "vehicle.state", { ...vehicleLive, airborne: true, in_flight: true });
}

async function openK(page: Page, seq: number): Promise<void> {
  await showView(page, "candidates");
  await page.locator(`[data-testid="candidate-card"][data-seq="${seq}"] button`).click();
  await expect(page.getByTestId("cand-review")).toBeVisible();
}

/** The open look the fake ground keeps, with `over` on top (pushed live). */
function pushLook(ground: FakeGround, over: Partial<Inspection>): Inspection {
  const now = ground.openLook();
  if (!now) throw new Error("no open look");
  const next = { ...now, ...over };
  ground.pushLive("candidates", "inspection.updated", next);
  return next;
}

/** Neither the page nor the review stage scrolls sideways. */
async function noSideScroll(page: Page, label: string): Promise<void> {
  const over = await page.evaluate(() => {
    const out: string[] = [];
    for (const el of [document.documentElement, ...document.querySelectorAll(".review-stage")])
      if (el.scrollWidth > el.clientWidth + 1) out.push(el.className || el.tagName);
    return out;
  });
  expect(over, label).toEqual([]);
}

const decisions = (ground: FakeGround) =>
  ground.calls(/^POST \/api\/v1\/candidates\/[^/]+\/decision$/);

test.describe("every size", () => {
  test("the pilot looks closer: question, strip, Gå lägre, countdown, Fortsätt söka nu", async ({
    page,
    ground,
  }) => {
    await flying(ground);
    await expect(page.getByTestId("op-badge")).toHaveText("ÖVNING");
    await openK(page, 3);

    // The question says what the drone will do; Avbryt sends nothing.
    await expect(page.getByTestId("cand-inspect")).toBeEnabled();
    await page.getByTestId("cand-inspect").click();
    const dialog = page.getByTestId("cand-inspect-dialog");
    await expect(dialog).toContainText("Titta närmare på K3?");
    await expect(dialog).toContainText(
      "sjunker så långt höjdreglerna tillåter (ÖVNING: rakt över)",
    );
    await expect(page.getByTestId("cand-inspect-nomap")).toContainText("Ingen höjdkarta");
    await expectSafetyVisible(page, "inspect question");
    await page.getByTestId("cand-inspect-yes").click();
    await expect.poll(() => decisions(ground).length).toBe(1);
    expect(decisions(ground)[0]?.body).toEqual({ decision: "inspect", note: "", subject_rev: 1 });
    await expect(dialog).toHaveCount(0);

    // The look flies: the strip, also on the card in the queue.
    const strip = page.getByTestId("inspect-strip");
    await expect(strip).toContainText("Tittar närmare på K3 · flyger dit");
    await expect(strip).toContainText("Överflygning tillåten i ÖVNING");
    const look = ground.openLook() as Inspection;
    ground.pushLive("candidates", "candidate.inspect_progress", {
      command_id: look.command_id,
      candidate_id: look.candidate_id,
      seq: 3,
      step: 2,
      phase: "step",
      above_ground_m: 32.4,
      floor_auto_above_ground_m: 30,
      floor_lowest_above_ground_m: 21,
      floor_reason: "map",
      laser_confirmed: false,
    });
    await expect(strip).toHaveText(
      /Tittar närmare på K3 · steg 2 · 32 m över mark · golv 21 m \(karta\) · laser: ingen/,
    );
    // Screen readers hear the step, not every new height (the strip changes at 2 Hz).
    const said = strip.getByTestId("inspect-strip-said");
    await expect(said).toHaveText("Tittar närmare på K3 · steg 2");
    await expect(said).toHaveAttribute("role", "status");
    await expect(strip).not.toHaveAttribute("role", "status");
    ground.pushLive("candidates", "candidate.inspect_progress", {
      command_id: look.command_id,
      candidate_id: look.candidate_id,
      seq: 3,
      step: 2,
      phase: "step",
      above_ground_m: 31.1,
      floor_auto_above_ground_m: 30,
      floor_lowest_above_ground_m: 21,
      floor_reason: "map",
      laser_confirmed: false,
    });
    await expect(strip).toContainText("31 m över mark");
    await expect(said).toHaveText("Tittar närmare på K3 · steg 2");
    await expect(page.getByTestId("cand-inspect")).toHaveCount(0);
    await expectSafetyVisible(page, "look flying");

    // "Gå lägre?": a toast wherever the pilot is, and the pilot's Ja goes to the ground.
    pushLook(ground, {
      phase: "lower_offer",
      lower_offer: {
        command_id: look.command_id,
        now_above_ground_m: 30,
        lowest_above_ground_m: 21,
        reason: "map",
      },
      lower_offer_until: inSeconds(58),
    });
    await expect(page.getByText("K3: drönaren frågar om den får gå lägre.")).toBeVisible();
    const lower = page.getByTestId("inspect-lower");
    await expect(lower).toContainText("Gå lägre, ned till 21 m över mark?");
    await expect(lower).toContainText(/svar inom 0:5\d/);
    await expectSafetyVisible(page, "go lower");
    await page.getByTestId("inspect-lower-yes").click();
    await expect
      .poll(() => ground.posts(`/api/v1/inspections/${look.command_id}/lower`).length)
      .toBe(1);
    await expect(lower).toHaveCount(0);

    // Done: the countdown, the buttons of the wait, Ta fler bilder.
    pushLook(ground, {
      phase: "await",
      done: {
        command_id: look.command_id,
        candidate_id: look.candidate_id,
        end: "clear",
        steps: 3,
        lowest_above_ground_m: 21,
        still_ids: [],
        clear: true,
      },
      await_until: inSeconds(103),
    });
    await expect(page.getByText("Titten på K3 är klar.")).toBeVisible();
    await expect(page.getByTestId("inspect-countdown")).toHaveText(
      /^Titten klar\. Beslut inom 1:4\d, annars fortsätter sökningen och K3 ligger kvar i kön\.$/,
    );
    await expect(page.getByTestId("inspect-end")).toContainText("Tydlig bild tagen · 3 steg");
    // The countdown ticks every second: not read out every second.
    await expect(page.getByTestId("inspect-await")).not.toHaveAttribute("aria-live", /.+/);
    await expect(page.getByTestId("cand-confirm")).toBeDisabled(); // no large view yet
    await expect(page.getByTestId("cand-reject")).toBeEnabled();
    await expect(page.getByTestId("cand-unsure")).toBeEnabled();
    await expectSafetyVisible(page, "waiting for the decision");
    await page.getByTestId("cand-more").click();
    await expect
      .poll(() => ground.posts("/api/v1/missions/M1/commands").map((r) => r.body))
      .toEqual([expect.objectContaining({ skill: "capture_stills", params: {} })]);

    // Fortsätt söka nu: the pilot's decision, back to the queue.
    await page.getByTestId("cand-resume").click();
    await expect.poll(() => decisions(ground).length).toBe(2);
    expect(decisions(ground)[1]?.body).toEqual({ decision: "resume", note: "", subject_rev: 2 });
    await expect(page.getByTestId("cand-review")).toHaveCount(0);
    await expect(page.locator('[data-testid="candidate-card"][data-seq="3"]')).toBeVisible();
    await expect(page.getByTestId("card-inspect")).toHaveCount(0);
  });

  test("a no from the gate stays in the question; another look greys the button", async ({
    page,
    ground,
  }) => {
    await flying(ground, { operation_mode: "live", overflight_allowed: false }, [card(3), card(4)]);
    await expect(page.getByTestId("op-badge")).toHaveText("SKARPT");
    await openK(page, 3);
    await page.getByTestId("cand-inspect").click();
    const dialog = page.getByTestId("cand-inspect-dialog");
    await expect(dialog).toContainText("tittar från sidan");
    ground.refuseInspect = {
      code: "FLOOR_ABOVE_CEILING",
      text: "Lägsta tillåtna höjd på platsen ligger över uppdragsramens tak.",
    };
    await page.getByTestId("cand-inspect-yes").click();
    await expect(page.getByTestId("cand-inspect-refused")).toHaveText(
      "Det går inte just nu: Lägsta tillåtna höjd på platsen ligger över uppdragsramens tak.",
    );
    await expect(dialog).toBeVisible();
    await page.keyboard.press("Escape");
    ground.refuseInspect = null;

    // A look on K4 is open: K3's button is grey and says why.
    ground.pushLive("candidates", "inspection.updated", {
      command_id: "0192f4a1-7b90-7a22-b1c3-100000000004",
      candidate_id: card(4).candidate_id,
      mission_id: "M1",
      seq: 4,
      started_at: new Date().toISOString(),
      started_by: "operator:demo",
      operation_mode: "live",
      overflight: false,
      phase: "flying",
    } satisfies Inspection);
    await expect(page.getByTestId("cand-inspect")).toBeDisabled();
    await expect(page.getByTestId("cand-inspect-why")).toHaveText("En annan titt pågår.");
  });

  test("an observer sees the look but neither starts it nor answers Gå lägre", async ({
    page,
    ground,
  }) => {
    ground.setSession(observerSession);
    await flying(ground);
    await openK(page, 3);
    await expect(page.getByTestId("cand-inspect")).toBeDisabled();
    await expect(page.getByTestId("cand-inspect-why")).toHaveText(
      "Bara piloten kan starta Titta närmare.",
    );
    ground.pushLive("candidates", "inspection.updated", {
      command_id: "0192f4a1-7b90-7a22-b1c3-100000000003",
      candidate_id: card(3).candidate_id,
      mission_id: "M1",
      seq: 3,
      started_at: new Date().toISOString(),
      started_by: "operator:demo",
      operation_mode: "exercise",
      overflight: true,
      phase: "lower_offer",
      lower_offer: {
        command_id: "0192f4a1-7b90-7a22-b1c3-100000000003",
        now_above_ground_m: 30,
        lowest_above_ground_m: 21,
        reason: "map",
      },
      lower_offer_until: inSeconds(50),
    } satisfies Inspection);
    await expect(page.getByTestId("inspect-lower")).toContainText(
      "Piloten avgör om drönaren får gå lägre.",
    );
    // The toast tells an observer that the pilot answers, not to answer.
    await expect(
      page.getByText("K3: drönaren frågar om den får gå lägre. Piloten svarar.").first(),
    ).toBeVisible();
    await expect(page.getByText("Svara under Granska")).toHaveCount(0);
    await expect(page.getByTestId("inspect-lower-yes")).toHaveCount(0);
    await expect(page.getByTestId("inspect-lower-no")).toHaveCount(0);
    await expect(page.getByTestId("cand-resume")).toHaveCount(0);
    await expect(page.getByTestId("cand-review")).toContainText("Under titten avgör piloten.");
  });

  test("the AI's suggestion is a question answered Ja or Nej", async ({ page, ground }) => {
    await flying(ground);
    await showView(page, "chat");
    ground.pushLive(
      "commands",
      "command.created",
      command({
        id: "K5",
        skill: "inspect_detection",
        params: { candidate_id: card(3).candidate_id },
        approval_id: "AP5",
      }),
    );
    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({
        id: "AP5",
        kind: "command",
        subject_id: "K5",
        summary_sv: "Titta närmare på K3 (hög prioritet, sedd i 2 bilder, 140 m NO)",
      }),
    );
    const ai = page.getByTestId("approval-AP5");
    await expect(ai).toContainText("AI:n föreslår: titta närmare");
    await expect(ai).toContainText(
      "Titta närmare på K3 (hög prioritet, sedd i 2 bilder, 140 m NO)",
    );
    await expect(ai.getByTestId("skill-inspect")).toBeVisible();
    await expect(ai.getByTestId("approve")).toHaveText("Ja");
    await expect(ai.getByTestId("reject")).toHaveText("Nej");
    await expectSafetyVisible(page, "AI suggestion");
    await ai.getByTestId("approve").click();
    await expect
      .poll(() => ground.posts("/api/v1/approvals/AP5/decision").map((r) => r.body))
      .toEqual([{ decision: "approve", subject_version: 1 }]);
  });

  test("Inställningar: the permit number first, then overflight in SKARPT", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await ground.resynced();
    await page.getByTestId("main-menu").click();
    await page.getByTestId("menu-settings").click();
    await expect(page.getByTestId("view-settings")).toBeVisible();
    await expect(page.getByTestId("permit-allow")).toBeDisabled();
    await expect(page.getByTestId("permit-save")).toBeDisabled();
    await page.getByTestId("permit-ref").fill("TS-0001");
    await page.getByTestId("permit-allow").check();
    await expectSafetyVisible(page, "settings");
    await page.getByTestId("permit-save").click();
    await expect(page.getByTestId("permit-msg")).toHaveText("Sparat.");
    expect(ground.calls(/^PUT \/api\/v1\/permit$/).map((r) => r.body)).toEqual([
      { specific_category_ref: "TS-0001", allow_overflight_live: true },
    ]);
  });

  test("an observer has no Inställningar", async ({ page, ground }) => {
    ground.setSession(observerSession);
    await ground.open();
    await ground.liveConnected();
    await page.getByTestId("main-menu").click();
    await expect(page.getByTestId("menu-status")).toBeVisible();
    await expect(page.getByTestId("menu-settings")).toHaveCount(0);
  });
});

test.describe("390×844", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

  test("the look and its wait fit a phone, STOP and FLY HOME in reach", async ({
    page,
    ground,
  }) => {
    await flying(ground);
    await openK(page, 3);
    await page.getByTestId("cand-inspect").click();
    await page.getByTestId("cand-inspect-yes").click();
    await expect(page.getByTestId("inspect-strip")).toBeVisible();
    await noSideScroll(page, "look flying");
    const look = ground.openLook() as Inspection;
    pushLook(ground, {
      phase: "lower_offer",
      lower_offer: {
        command_id: look.command_id,
        now_above_ground_m: 30,
        lowest_above_ground_m: 21,
        reason: "map",
      },
      lower_offer_until: inSeconds(60),
    });
    await expect(page.getByTestId("inspect-lower")).toBeVisible();
    await noSideScroll(page, "go lower");
    await expectSafetyVisible(page, "phone go lower");
    // Nej goes to the drone at once (it stops asking and stays at its height).
    await page.getByTestId("inspect-lower-no").click();
    await expect
      .poll(() => ground.posts(`/api/v1/inspections/${look.command_id}/stay`).length)
      .toBe(1);
    expect(ground.posts(`/api/v1/inspections/${look.command_id}/lower`)).toEqual([]);
    await expect(page.getByTestId("inspect-lower")).toHaveCount(0);
    pushLook(ground, {
      phase: "await",
      lower_offer: null,
      lower_offer_until: null,
      done: {
        command_id: look.command_id,
        candidate_id: look.candidate_id,
        end: "lower_declined",
        steps: 2,
        lowest_above_ground_m: 30,
        still_ids: [],
        clear: false,
      },
      await_until: inSeconds(120),
    });
    await expect(page.getByTestId("inspect-await")).toContainText("Stannade på 30 m över mark");
    await noSideScroll(page, "waiting");
    await expectSafetyVisible(page, "phone waiting");
  });
});
