import { useEffect, useMemo, useSyncExternalStore } from "react";
import type { VehicleAction } from "../api/paths";
import { actionBus } from "../state/actionBus";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { type ActionSnapshot, VehicleActionMachine } from "./vehicleAction";

export interface UseVehicleAction extends ActionSnapshot {
  trigger(): void;
  dismiss(): void;
}

export function useVehicleAction(
  vehicleId: string | null,
  action: VehicleAction,
): UseVehicleAction {
  const machine = useMemo(
    () =>
      new VehicleActionMachine(vehicleId, action, {
        fetch: (input, init) => fetch(input, init),
        log: (level, message) => useUiStore.getState().log(level, message),
        onUnauthorized: () => {
          const s = useSessionStore.getState();
          if (s.status === "ready") s.set({ needsReauth: true });
        },
      }),
    [vehicleId, action],
  );
  useEffect(() => actionBus.subscribe((o) => machine.outcome(o)), [machine]);
  useEffect(() => () => machine.destroy(), [machine]);
  const snap = useSyncExternalStore(
    (cb) => machine.subscribe(cb),
    () => machine.snapshot,
  );
  return { ...snap, trigger: () => machine.trigger(), dismiss: () => machine.dismiss() };
}
