// Granska: the candidate queue and a person's decision. A candidate arrives live and blinks;
// the low ones wait under a count; Bekräfta stays grey and the AI's text folded until a
// picture has been opened large; a flagged claim is marked unverified; the confirmation is a
// hold and a question, then the position in three forms, the find number and the toast.
// On a phone nothing scrolls sideways. The candidates sit at the public simulator site.
import type { Page } from "@playwright/test";
import type { CandidateCard, CandidateDetail } from "@searchdrone/schemas";
import { panelConfig } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, holdClick, showView } from "../helpers/safety";

const AT = { lat: 59.28842, lon: 18.15953 };

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
    fields: { "garment:0": "match", hair: "not_visible" },
    assessed_profile_version: 1,
    current_profile_version: 1,
    ...over,
  };
}

function extra(seq: number): Pick<CandidateDetail, "observations" | "assessments"> {
  const c = card(seq);
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
    assessments: [
      {
        id: `a-${seq}`,
        candidate_id: c.candidate_id,
        created_at: new Date().toISOString(),
        engine: "strong",
        explanation_sv: "En person i röd jacka sitter vid stenen.",
        fields: { "garment:0": "match" },
        flags: ["motion_claim"],
        latency_ms: 900,
        model_id: "test",
        overall: 0.74,
        person: "yes",
        person_p: 0.91,
        profile_version: 1,
      },
    ],
  };
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

// These run in every project (1280×720 desktop, tablet, the 360 and 375 px phones), so
// STOP and FLY HOME are checked with each new overlay open at every mandated size.
test.describe("every size", () => {
  test("a person confirms a find only after looking at the picture", async ({ page, ground }) => {
    ground.setConfig({ features: { ...panelConfig.features, candidates: true } });
    ground.candidates = [card(2, { priority: 0.1, priority_class: "low" })];
    ground.candidateDetail[card(3).candidate_id] = extra(3);
    await ground.open();
    await ground.liveConnected();
    await ground.resynced();
    await showView(page, "candidates");

    // A new card arrives live and blinks once; the low one waits under its count.
    ground.pushLive("candidates", "candidate.new", card(3));
    const k3 = page.locator('[data-testid="candidate-card"][data-seq="3"]');
    await expect(k3).toHaveAttribute("data-fresh", "true");
    await expect(k3).toHaveAttribute("data-fresh", "false");
    await expect(k3).toContainText("Moln stark");
    await expect(k3).toContainText("sedd i 2 av 3 bilder");
    await expect(page.getByTestId("review-badge")).toHaveText("Att granska: 1");
    await expect(page.getByTestId("main-menu")).toBeVisible();
    await expectSafetyVisible(page, "queue with badge");
    await noSideScroll(page, "queue");
    await expect(page.locator('[data-testid="candidate-card"][data-seq="2"]')).toHaveCount(0);
    await expect(page.getByTestId("cand-low-toggle")).toHaveText("Låg prioritet (1)");
    await page.getByTestId("cand-low-toggle").click();
    await expect(page.locator('[data-testid="candidate-card"][data-seq="2"]')).toBeVisible();

    // Open K3: Bekräfta grey, the AI folded, Titta närmare not there yet.
    await k3.locator("button").click();
    const review = page.getByTestId("cand-review");
    await expect(review).toBeVisible();
    await expect(page.getByTestId("cand-confirm")).toBeDisabled();
    await expect(page.getByTestId("cand-ai-toggle")).toBeDisabled();
    await expect(page.getByTestId("cand-inspect")).toBeDisabled();
    await expect(review).not.toContainText("röd jacka");
    await expectSafetyVisible(page, "review");
    await noSideScroll(page, "review");

    // A large picture opens both.
    await page.getByTestId("cand-larger").click();
    await expect(page.getByTestId("cand-large")).toBeVisible();
    await expectSafetyVisible(page, "large picture open");
    await page.getByTestId("cand-large").getByRole("button").click();
    await page.getByTestId("cand-ai-toggle").click();
    await expect(review).toContainText("En person i röd jacka sitter vid stenen.");
    await expect(page.getByTestId("cand-ai-flag")).toContainText("AI-påstående, ej verifierat");
    await expect(page.getByTestId("cand-confirm")).toBeEnabled();

    // Hold, then answer the question. Centred, so a phone's sheet is not over it.
    await page.getByTestId("cand-confirm").evaluate((el) => el.scrollIntoView({ block: "center" }));
    await holdClick(page, "cand-confirm");
    await expect(page.getByTestId("cand-confirm-dialog")).toContainText(
      "Bekräfta K3 som personen ni söker?",
    );
    await expectSafetyVisible(page, "confirm dialog open");
    await page.getByTestId("cand-confirm-yes").click();

    const post = ground.calls(/^POST \/api\/v1\/candidates\/[^/]+\/decision$/);
    expect(post).toHaveLength(1);
    expect(post[0]?.body).toEqual({ decision: "confirm", note: "", subject_rev: 1 });
    const found = page.getByTestId("cand-found");
    await expect(found).toContainText("Fynd F1: K3");
    await expect(page.getByTestId("cand-coord-decimal")).toHaveText("59.288420, 18.159530");
    await expect(page.getByTestId("cand-coord-dms")).toHaveText("59°17′18.3″N 18°09′34.3″E");
    await expect(page.getByTestId("cand-coord-sweref")).toHaveText("N 6576436 E 679958");
    await expect(page.getByTestId("cand-confirmed-by")).toHaveText(/^Bekräftad av dig \d\d:\d\d$/);
    await expect(page.getByTestId("cand-gpx")).toHaveAttribute(
      "href",
      `/api/v1/candidates/${card(3).candidate_id}/export.gpx`,
    );
    await expect(page.getByText("Fynd bekräftat: K3")).toBeVisible();
    await expect(page.getByTestId("review-badge")).toHaveCount(0);
    await expectSafetyVisible(page, "find confirmed");

    // The pilot's Ångra asks for a reason first; cancelling it leaves the find.
    await page.getByTestId("cand-undo").click();
    await expect(page.getByTestId("cand-undo-yes")).toBeDisabled();
    await expectSafetyVisible(page, "undo dialog open");
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("cand-undo-dialog")).toHaveCount(0);

    // Back in Flyg the mission column names the find.
    await page.getByTestId("cand-back").click();
    if ((await page.getByTestId("mode-fly").count()) > 0)
      await page.getByTestId("mode-fly").click();
    else await page.getByTestId("review-back").click();
    const missionTab = page.getByTestId("side-tab-mission");
    if ((await missionTab.count()) > 0) await missionTab.click();
    await expect(page.getByTestId("finds-confirmed")).toHaveText("Bekräftat: F1 (K3)");

    // And the map marks it F1 (where this browser has WebGL2 for the map).
    const labels = async (): Promise<string[] | null> =>
      page.evaluate(async () => {
        const map = (
          window as unknown as {
            __sd?: { map?: { getSource(id: string): { getData(): Promise<unknown> } | undefined } };
          }
        ).__sd?.map;
        const src = map?.getSource("candidates");
        if (!src) return null;
        const fc = (await src.getData()) as {
          features: { geometry: { type: string }; properties: { label: string } }[];
        };
        return fc.features
          .filter((f) => f.geometry.type === "Point")
          .map((f) => f.properties.label);
      });
    const webgl2 = await page.evaluate(
      () => document.createElement("canvas").getContext("webgl2") !== null,
    );
    if (webgl2) await expect.poll(labels).toEqual(["K2", "F1"]);
  });

  test("the copy button puts all three forms on the clipboard", async ({
    page,
    ground,
    context,
    browserName,
  }) => {
    test.skip(browserName !== "chromium", "clipboard permissions are Chromium's");
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    ground.setConfig({ features: { ...panelConfig.features, candidates: true } });
    ground.candidates = [card(4, { state: "confirmed", rev: 2 })];
    ground.candidateDetail[card(4).candidate_id] = extra(4);
    await ground.open();
    await ground.resynced();
    await showView(page, "candidates");
    await page.getByTestId("cand-tab-confirmed").click();
    await page.locator('[data-testid="candidate-card"][data-seq="4"] button').click();
    await page.getByTestId("cand-copy").click();
    await expect(page.getByTestId("cand-found")).toContainText("Kopierat");
    const text = await page.evaluate(() => navigator.clipboard.readText());
    expect(text).toContain("59.288420, 18.159530");
    expect(text).toContain("59°17′18.3″N 18°09′34.3″E");
    expect(text).toContain("SWEREF 99 TM: N 6576436 E 679958");
  });
});

// The size the panel is designed for on a phone, in addition to the projects' sizes.
test.describe("390×844", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

  test("the queue and the review fit a phone without sideways scrolling", async ({
    page,
    ground,
  }) => {
    ground.setConfig({ features: { ...panelConfig.features, candidates: true } });
    ground.candidates = [card(5), card(6, { priority: 0.2, priority_class: "low" })];
    ground.candidateDetail[card(5).candidate_id] = extra(5);
    await ground.open();
    await ground.liveConnected();
    await ground.resynced();
    await page.getByTestId("review-badge").click();
    await expect(page.getByTestId("view-candidates")).toBeVisible();
    await noSideScroll(page, "queue");
    await page.getByTestId("cand-low-toggle").click();
    await noSideScroll(page, "low list");
    await page.locator('[data-testid="candidate-card"][data-seq="5"] button').click();
    await expect(page.getByTestId("cand-review")).toBeVisible();
    await noSideScroll(page, "review");
    await expectSafetyVisible(page, "phone review");
  });
});
