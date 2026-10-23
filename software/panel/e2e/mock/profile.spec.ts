// The search profile on a phone (390×844): the AI's proposal arrives, the pilot corrects a
// colour, a garment and a certainty on the card and approves it; an observer sees the
// same card without the button. The safety bar stays visible over it all. The person is
// fictional.
import type { Page } from "@playwright/test";
import { MISSION, observerSession } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, showView } from "../helpers/safety";

test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

const PROPOSAL = {
  profile_id: "PR1",
  profile_version: 1,
  mission_id: MISSION,
  approved_by: null,
  approved_at: null,
  subject_category: "dementia",
  garments: [
    { part: "outer", colors: ["red"], certainty: "certain" },
    { part: "lower", colors: ["blue"], certainty: "unsure" },
  ],
  items: [],
  last_seen: { lat: -35.3632, lon: 149.1652, at_utc: "2026-01-01T10:00:00Z" },
  free_text: "",
};

async function openProfile(page: Page): Promise<void> {
  await showView(page, "map");
  const tab = page.getByTestId("profile-tab");
  if ((await tab.count()) > 0) await tab.click(); // without WebGL the card is on the page
  await expect(page.getByTestId("profile-card")).toBeVisible();
}

test("the pilot corrects the AI's profile and approves it", async ({ page, ground }) => {
  ground.profile = { approved: null, proposal: PROPOSAL, proposal_id: "X1", proposed_by: "ai" };
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  await openProfile(page);
  const card = page.getByTestId("profile-card");
  await expect(card.getByTestId("profile-status")).toHaveText("Förslag, inte godkänt");
  await expect(card.getByTestId("profile-free-warning")).toContainText("utan datainsamling");

  await card.getByTestId("profile-garment-0-color").selectOption("green");
  await card.getByTestId("profile-garment-1-part").selectOption("full_body");
  await card.getByTestId("profile-garment-1-certainty").selectOption("likely");
  await expectSafetyVisible(page);
  await card.getByTestId("profile-approve").click();

  await expect(card.getByTestId("profile-status")).toHaveText("Godkänd, version 1");
  const post = ground.calls(/^POST \/api\/v1\/missions\/[^/]+\/search-profile$/);
  expect(post).toHaveLength(1);
  const sent = post[0]?.body as {
    profile: { garments: Record<string, unknown>[]; approved_by?: unknown };
    base_version: number;
    proposal_id: string | null;
  };
  expect(sent).toMatchObject({ base_version: 0, proposal_id: "X1" });
  expect(sent.profile.garments[0]).toMatchObject({ part: "outer", colors: ["green"] });
  expect(sent.profile.garments[1]).toMatchObject({ part: "full_body", certainty: "likely" });
  expect(sent.profile).not.toHaveProperty("approved_by");
  await expectSafetyVisible(page);
});

test("a new proposal on the live stream shows on the open card", async ({ page, ground }) => {
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  await openProfile(page);
  await expect(page.getByTestId("profile-status")).toHaveText("Ingen profil");
  ground.profile = { approved: null, proposal: PROPOSAL, proposal_id: "X1", proposed_by: "ai" };
  // The event carries ids only; the card fetches the profile itself.
  ground.pushLive("search_profile", "search_profile.proposed", {
    row_id: "X1",
    profile_id: "PR1",
    profile_version: 1,
  });
  await expect(page.getByTestId("profile-garment-0")).toBeVisible();
});

test("a new proposal while the pilot edits keeps the edits", async ({ page, ground }) => {
  ground.profile = { approved: null, proposal: PROPOSAL, proposal_id: "X1", proposed_by: "ai" };
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  await openProfile(page);
  const card = page.getByTestId("profile-card");
  await card.getByTestId("profile-garment-0-color").selectOption("green");
  ground.profile = { ...ground.profile, proposal_id: "X2" };
  ground.pushLive("search_profile", "search_profile.proposed", {
    row_id: "X2",
    profile_id: "PR1",
    profile_version: 1,
  });
  await expect(card.getByTestId("profile-newer")).toBeVisible();
  await expect(card.getByTestId("profile-garment-0-color")).toHaveValue("green");
  await card.getByTestId("profile-show-newer").click();
  await expect(card.getByTestId("profile-garment-0-color")).toHaveValue("red");
  await expect(card.getByTestId("profile-newer")).toHaveCount(0);
});

test("an observer sees the card but not the button", async ({ page, ground }) => {
  ground.setSession(observerSession);
  ground.profile = { approved: null, proposal: PROPOSAL, proposal_id: "X1", proposed_by: "ai" };
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  await openProfile(page);
  const card = page.getByTestId("profile-card");
  await expect(card.getByTestId("profile-garment-0-color")).toBeDisabled();
  await expect(card.getByTestId("profile-approve")).toHaveCount(0);
  await expect(card).toContainText("Bara piloten kan ändra och godkänna sökprofilen.");
  expect(ground.calls(/^POST \/api\/v1\/missions\/[^/]+\/search-profile$/)).toHaveLength(0);
  await expectSafetyVisible(page);
});

// The open card never covers the safety bar, at the desktop and the reference phone size.
for (const [width, height] of [
  [1280, 720],
  [375, 812],
] as const) {
  test.describe(`profile card open at ${width}×${height}`, () => {
    const phone = width < 600;
    test.use({ viewport: { width, height }, isMobile: phone, hasTouch: phone });

    test("the safety bar stays visible", async ({ page, ground }) => {
      ground.profile = { approved: null, proposal: PROPOSAL, proposal_id: "X1", proposed_by: "ai" };
      await ground.open();
      await ground.liveConnected();
      await ground.resynced();
      await openProfile(page);
      await expectSafetyVisible(page);
    });
  });
}
