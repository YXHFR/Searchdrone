// The probability map's score per sector (POA left and PSR), read again on every new map
// version. Only a ranking for the people and the AI: nothing here starts a flight.
import { useEffect, useState } from "react";
import { api } from "../api/http";
import { paths } from "../api/paths";
import type { SectorScore } from "../api/types";
import { useMissionStore } from "../state/missionStore";

const two = new Intl.NumberFormat("sv-SE", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const three = new Intl.NumberFormat("sv-SE", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});

/** "0,08" and "0,002": the sector list's figures, with Swedish decimal commas. */
export function scoreText(s: SectorScore): { poa: string; psr: string } {
  return { poa: two.format(s.poa), psr: three.format(s.psr) };
}

/** The scores by sector id; empty before the first map (or without a mission). */
export function useSectorScores(missionId: string | null): Record<string, SectorScore> {
  const rev = useMissionStore((s) => s.mapRev);
  const [scores, setScores] = useState<Record<string, SectorScore>>({});
  // biome-ignore lint/correctness/useExhaustiveDependencies: rev is the refetch signal
  useEffect(() => {
    if (!missionId) {
      setScores({});
      return;
    }
    let stale = false;
    api<SectorScore[]>(paths.sectorPoa(missionId))
      .then((list) => {
        if (!stale) setScores(Object.fromEntries(list.map((s) => [s.sector_id, s])));
      })
      .catch(() => {
        // the list keeps the figures it had
      });
    return () => {
      stale = true;
    };
  }, [missionId, rev]);
  return scores;
}
