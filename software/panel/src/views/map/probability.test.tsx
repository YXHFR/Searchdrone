import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ProbabilityMapView, SectorScore } from "../../api/types";
import { useMissionStore } from "../../state/missionStore";

const GRID: ProbabilityMapView["grid"] = {
  ipp_lat: -35.363261,
  ipp_lon: 149.16523,
  x0_m: -800,
  y0_m: -800,
  nx: 64,
  ny: 64,
  cell_m: 25,
  west: 149.156,
  south: -35.3705,
  east: 149.1745,
  north: -35.356,
};
let version = 1;
let noMap = false;
const calls: string[] = [];
// Reset after every test, so a failed assertion cannot leave the flag set for the next.
afterEach(() => {
  noMap = false;
});
vi.mock("../../api/http", () => ({
  HttpError: class extends Error {},
  api: async (path: string) => {
    calls.push(path);
    if (path.endsWith("/sectors/poa"))
      return [
        {
          sector_id: "A",
          poa: 0.0812,
          covered_pct: 100,
          mean_pod: 0.71,
          pos_plan: 0.05,
          est_min: 12,
          psr: 0.0042,
          spacing_m: 41.6,
        } satisfies SectorScore,
      ];
    if (noMap) return null;
    return {
      version,
      reason: "prior",
      grid: GRID,
      row: 0.123,
      frozen: false,
      event_seq: 0,
      profile_version: 1,
      notes: ["no_terrain", "child_statistics"],
      mission_id: "m1",
      p_b64: "",
      pod_b64: "",
      created_at: "2026-10-23T10:00:00Z",
    } satisfies ProbabilityMapView;
  },
}));

const { corners, rowPercent, useProbabilityLayers } = await import("./probability");
const { ProbabilityLegend } = await import("./ProbabilityLegend");
const { scoreText, useSectorScores } = await import("../../mission/sectorScores");

describe("probability map helpers", () => {
  it("puts the image's corners NW, NE, SE, SW", () => {
    expect(corners(GRID)).toEqual([
      [GRID.west, GRID.north],
      [GRID.east, GRID.north],
      [GRID.east, GRID.south],
      [GRID.west, GRID.south],
    ]);
  });

  it("says the rest of the world in whole percent", () => {
    expect(rowPercent(0.123)).toBe("12 %");
    expect(rowPercent(1.5)).toBe("100 %");
    expect(rowPercent(-1)).toBe("0 %");
  });

  it("writes the sector figures with Swedish decimal commas", () => {
    expect(scoreText({ poa: 0.0812, psr: 0.0042 } as SectorScore)).toEqual({
      poa: "0,08",
      psr: "0,004",
    });
  });
});

describe("the legend", () => {
  it("shows the scale, the share outside the map and the notes", () => {
    render(
      <ProbabilityLegend
        meta={{ version: 1, row: 0.12, frozen: false, notes: ["no_terrain"], grid: GRID }}
        poa
        pod={false}
      />,
    );
    expect(screen.getByText("Sannolikhet")).toBeTruthy();
    expect(screen.getByText("låg → hög")).toBeTruthy();
    expect(screen.queryByText("Sökt (POD)")).toBeNull();
    expect(screen.getByTestId("prob-row").textContent).toBe("Utanför kartan: 12 %");
    expect(screen.getByTestId("prob-note-no_terrain").textContent).toBe(
      "Ingen terrängdata, bara avståndsringar.",
    );
  });

  it("says when the map is locked", () => {
    render(
      <ProbabilityLegend
        meta={{ version: 3, row: 0.2, frozen: true, notes: [], grid: GRID }}
        poa
        pod
      />,
    );
    expect(screen.getByTestId("prob-frozen")).toBeTruthy();
    expect(screen.getByText("Sökt (POD)")).toBeTruthy();
  });
});

describe("the layers follow the newest version", () => {
  function fakeMap() {
    const sources: Record<string, { url: string; updateImage: (o: { url: string }) => void }> = {};
    const layers: Record<string, { visibility?: string; before?: string | undefined }> = {};
    return {
      sources,
      layers,
      map: {
        getSource: (id: string) => sources[id],
        addSource: (id: string, s: { url: string }) => {
          sources[id] = {
            url: s.url,
            updateImage(o) {
              this.url = o.url;
            },
          };
        },
        getLayer: (id: string) => layers[id],
        addLayer: (l: { id: string }, before?: string) => {
          layers[l.id] = { before };
        },
        setLayoutProperty: (id: string, _k: string, v: string) => {
          const l = layers[id];
          if (l) l.visibility = v;
        },
      },
    };
  }

  it("adds the heat layer on and the sought layer off, then refetches on an event", async () => {
    const f = fakeMap();
    const handle = { current: { map: f.map, mod: {} } } as never;
    const shown = { poa: true, pod: false };
    const { result } = renderHook(() => useProbabilityLayers(true, handle, "m1", shown));
    await waitFor(() => expect(result.current?.version).toBe(1));
    expect(f.layers["prob-poa"]?.visibility).toBe("visible");
    expect(f.layers["prob-pod"]?.visibility).toBe("none");
    expect(f.sources["prob-poa"]?.url).toBe(
      "/api/v1/missions/m1/probability-map.png?layer=poa&v=1",
    );
    version = 2;
    act(() => useMissionStore.getState().bumpMap());
    // The old version stays drawn while the new one loads: no blink.
    expect(result.current?.version).toBe(1);
    expect(f.layers["prob-poa"]?.visibility).toBe("visible");
    await waitFor(() => expect(result.current?.version).toBe(2));
    expect(f.sources["prob-poa"]?.url).toContain("&v=2");
  });

  it("draws nothing before the first map (the ground answers null)", async () => {
    noMap = true;
    const before = calls.length;
    const f = fakeMap();
    const handle = { current: { map: f.map, mod: {} } } as never;
    const { result } = renderHook(() =>
      useProbabilityLayers(true, handle, "m2", { poa: true, pod: true }),
    );
    await waitFor(() => expect(calls.length).toBe(before + 1));
    expect(result.current).toBeNull();
    expect(f.layers["prob-poa"]).toBeUndefined();
  });

  it("reads the sector scores again on a new version", async () => {
    const before = calls.filter((c) => c.endsWith("/sectors/poa")).length;
    const { result } = renderHook(() => useSectorScores("m1"));
    await waitFor(() => expect(result.current.A?.poa).toBe(0.0812));
    act(() => useMissionStore.getState().bumpMap());
    await waitFor(() =>
      expect(calls.filter((c) => c.endsWith("/sectors/poa")).length).toBe(before + 2),
    );
  });
});
