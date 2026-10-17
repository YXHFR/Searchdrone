// The panel against the whole stack on ArduCopter SITL with the scripted commander:
// step 1 of the demo from the panel (PS-2), STOP in the air and Resume (PS-3), and FLY
// HOME with the AI process stopped (PS-4). Video (PS-1) and the ground-station restart
// (PS-5) are not covered here.
import { expect, type Page, test } from "@playwright/test";
import { stackUrl, stopBrain } from "./stack";

interface Hooks {
  liveStore: {
    getState(): Record<string, unknown> & {
      telemetry: Tel | null;
      vehicle: Veh | null;
      mission: Mission | null;
    };
  };
  missionStore: { getState(): { sectors: { properties: { progress: number } }[] } };
}
interface Tel {
  armed: boolean;
  groundspeed_ms: number;
  lat: number;
  lon: number;
  home: { lat: number; lon: number } | null;
  landed_state: string;
}
interface Veh {
  snapshot: { executor_state: string } | null;
}
interface Mission {
  id: string;
  status: string;
}

const live = <T>(page: Page, pick: string): Promise<T> =>
  page.evaluate((p) => {
    const s = (window as unknown as { __sd: Hooks }).__sd.liveStore.getState() as Record<
      string,
      unknown
    >;
    return p
      .split(".")
      .reduce<unknown>((o, k) => (o == null ? o : (o as Record<string, unknown>)[k]), s);
  }, pick) as Promise<T>;

const sectorProgress = (page: Page): Promise<number> =>
  page.evaluate(() => {
    const s = (window as unknown as { __sd: Hooks }).__sd.missionStore.getState().sectors;
    return s.length === 0 ? 0 : Math.min(...s.map((x) => x.properties.progress));
  });

function metres(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const k = 111_320;
  return Math.hypot((a.lat - b.lat) * k, (a.lon - b.lon) * k * Math.cos((a.lat * Math.PI) / 180));
}

async function say(page: Page, text: string): Promise<void> {
  const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
  await box.fill(text);
  await box.press("Enter");
}

/** A new mission to READY: the request in the chat, the AI's charter card, approval. */
async function readyMission(page: Page): Promise<void> {
  await page.goto(stackUrl());
  await expect(page.getByTestId("mission-title")).toContainText("Uppdrag", { timeout: 30_000 });
  // A mission left READY by an earlier test is used as it is; a landed one closes by itself
  // and the panel opens a new one.
  await expect
    .poll(() => live<string>(page, "mission.status"), { timeout: 60_000 })
    .toMatch(/^(DRAFT|READY)$/);
  if ((await live<string>(page, "mission.status")) === "READY") return;
  await say(page, "Sök av rutan norr om hem.");
  const card = page.locator('[data-kind="charter"]').last();
  await expect(card.getByTestId("approve")).toBeEnabled({ timeout: 30_000 });
  await expect(card.getByRole("radio", { name: /B – AI:n flyger fritt/ })).toBeChecked();
  await card.getByTestId("approve").click();
  await untilStatus(page, "READY", 90_000);
}

/** The panel's mission status reaches `want`; on a timeout say what the server has. */
async function untilStatus(page: Page, want: string, limitMs: number): Promise<void> {
  const until = Date.now() + limitMs;
  while (Date.now() < until) {
    if ((await live<string>(page, "mission.status")) === want) return;
    await page.waitForTimeout(500);
  }
  const id = await live<string>(page, "mission.id");
  const server = await page.evaluate(async (mid) => {
    const r = await fetch(`/api/v1/missions/${mid}`);
    const d = (await r.json()) as {
      mission?: { status?: string };
      charter?: { status?: string; install_status?: string };
    };
    return `${d.mission?.status} charter=${d.charter?.status}/${d.charter?.install_status}`;
  }, id);
  throw new Error(
    `mission ${id}: panel ${await live<string>(page, "mission.status")}, server ${server}, ws ${await live<string>(page, "ws")}`,
  );
}

async function startFlight(page: Page): Promise<void> {
  const cc = page.locator(".chat").getByTestId("clearance");
  await expect(cc.getByRole("checkbox")).toBeVisible({ timeout: 90_000 });
  await cc.getByRole("checkbox").check();
  await cc.getByTestId("clearance-button").click();
  await expect(page.locator(".chat").getByTestId("clearance")).toContainText("Klartecken gäller");
  await say(page, "Starta");
  await expect(page.getByTestId("pill-vehicle")).toHaveText("Lyfter", { timeout: 60_000 });
}

test("steg 1 genom panelen, STOPP i luften och Fortsätt (PS-2, PS-3)", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await readyMission(page);
  await startFlight(page);
  await expect(page.getByTestId("pill-vehicle")).toHaveText("Söker", { timeout: 120_000 });

  // PS-3: STOP during the search; acknowledged within 1 s, the vehicle stops.
  await page.getByTestId("safety-stop").click();
  await expect(page.getByTestId("safety-stop")).toContainText("Kvitterat", { timeout: 1_000 });
  await expect
    .poll(() => live<number>(page, "telemetry.groundspeed_ms"), { timeout: 5_000 })
    .toBeLessThan(0.5);
  const row = page.locator(".chat").getByTestId("resume-row");
  await expect(row).toBeVisible({ timeout: 5_000 });
  await page.waitForTimeout(2_100); // Resume is armed after 2 s
  await row.getByTestId("resume").click();
  await expect
    .poll(() => live<string>(page, "vehicle.snapshot.executor_state"), { timeout: 15_000 })
    .not.toBe("HELD");

  // PS-2: the search finishes, the AI flies home and lands.
  await expect
    .poll(() => sectorProgress(page), { timeout: 360_000, intervals: [2_000] })
    .toBeGreaterThanOrEqual(100);
  await expect(page.getByTestId("pill-vehicle")).toHaveText("På marken", { timeout: 240_000 });
  await expect.poll(() => live<boolean>(page, "telemetry.armed"), { timeout: 60_000 }).toBe(false);
  expect(errors).toEqual([]);
});

test("FLYG HEM med AI:n avstängd (PS-4)", async ({ page }) => {
  await readyMission(page);
  await startFlight(page);
  await expect(page.getByTestId("pill-vehicle")).toHaveText(/Flyger till punkt|Söker/, {
    timeout: 120_000,
  });

  stopBrain();
  await expect(page.getByTestId("brain-row")).toContainText("AI:n är nere", { timeout: 6_000 });
  await expect(page.getByTestId("ai-down-banner")).toBeVisible({ timeout: 6_000 });

  await page.getByTestId("safety-home").click();
  await expect(page.getByTestId("safety-home")).toContainText("Kvitterat", { timeout: 5_000 });
  const deadline = Date.now() + 240_000;
  while (Date.now() < deadline) {
    await expect(page.getByTestId("resume")).toHaveCount(0);
    const tel = await live<Tel | null>(page, "telemetry");
    if (tel && !tel.armed) break;
    await page.waitForTimeout(1_000);
  }
  const tel = await live<Tel | null>(page, "telemetry");
  expect(tel?.armed).toBe(false);
  if (!tel?.home) throw new Error("no home position in the telemetry");
  expect(metres(tel, tel.home)).toBeLessThan(5);
});
