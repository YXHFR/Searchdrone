import { type PointerEvent, useRef, useState } from "react";
import { stickFromPointer } from "./sticks";

interface Props {
  label: string;
  disabled: boolean;
  onChange(pos: { x: number; y: number }): void;
}

/** On-screen stick. Springs back to the centre when released. */
export function Stick({ label, disabled, onChange }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  function update(e: PointerEvent<HTMLDivElement>): void {
    const el = ref.current;
    if (!el || disabled) return;
    const r = el.getBoundingClientRect();
    const p = stickFromPointer(
      e.clientX - (r.left + r.width / 2),
      e.clientY - (r.top + r.height / 2),
      r.width / 2,
    );
    setPos(p);
    onChange(p);
  }
  function release(): void {
    setPos({ x: 0, y: 0 });
    onChange({ x: 0, y: 0 });
  }

  return (
    <div className="stick-wrap">
      <div
        ref={ref}
        className="stick"
        role="img"
        aria-label={label}
        aria-disabled={disabled}
        data-disabled={disabled}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          update(e);
        }}
        onPointerMove={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) update(e);
        }}
        onPointerUp={release}
        onPointerCancel={release}
      >
        <div
          className="stick-knob"
          style={{ transform: `translate(${pos.x * 50}%, ${-pos.y * 50}%)` }}
        />
      </div>
      <span className="stick-label">{label}</span>
    </div>
  );
}
