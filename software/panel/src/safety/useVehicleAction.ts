import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { VehicleAction } from "../api/paths";
import { actionBus } from "../state/actionBus";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { type ActionSnapshot, VehicleActionMachine } from "./vehicleAction";

export interface UseVehicleAction extends ActionSnapshot {
  trigger(): void;
  dismiss(): void;
}

/**
 * One machine for the lifetime of the button. A changing vehicle id (for example config
 * arriving after a press on the emergency page) only affects the NEXT press; a press in
 * flight keeps its target, its retries and its status.
 */
export function useVehicleAction(
  vehicleId: string | null,
  action: VehicleAction,
): UseVehicleAction {
  const idRef = useRef(vehicleId);
  idRef.current = vehicleId;
  const [machine] = useState(
    () =>
      new VehicleActionMachine(() => idRef.current, action, {
        fetch: (input, init) => fetch(input, init),
        log: (level, message) => useUiStore.getState().log(level, message),
        onUnauthorized: () => {
          const s = useSessionStore.getState();
          if (s.status === "ready") s.set({ needsReauth: true });
        },
      }),
  );
  useEffect(() => actionBus.subscribe((o) => machine.outcome(o)), [machine]);
  const snap = useSyncExternalStore(
    (cb) => machine.subscribe(cb),
    () => machine.snapshot,
  );
  return { ...snap, trigger: () => machine.trigger(), dismiss: () => machine.dismiss() };
}
