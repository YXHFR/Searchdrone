import { useEffect, useState } from "react";

/** Name of the first connected game controller (Gamepad API), or null. */
export function useGamepadName(): string | null {
  const [name, setName] = useState<string | null>(null);
  useEffect(() => {
    if (typeof navigator.getGamepads !== "function") return;
    const read = (): void => {
      const pad = Array.from(navigator.getGamepads()).find((p) => p !== null) ?? null;
      setName(pad ? pad.id : null);
    };
    read();
    window.addEventListener("gamepadconnected", read);
    window.addEventListener("gamepaddisconnected", read);
    return () => {
      window.removeEventListener("gamepadconnected", read);
      window.removeEventListener("gamepaddisconnected", read);
    };
  }, []);
  return name;
}
