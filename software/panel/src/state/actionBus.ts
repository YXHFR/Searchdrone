import type { ActionOutcome } from "../api/types";

type Fn = (o: ActionOutcome) => void;
const listeners = new Set<Fn>();

/** WS `vehicle.action` events fan out to every safety-button machine. */
export const actionBus = {
  publish(o: ActionOutcome): void {
    for (const fn of listeners) fn(o);
  },
  subscribe(fn: Fn): () => void {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};
