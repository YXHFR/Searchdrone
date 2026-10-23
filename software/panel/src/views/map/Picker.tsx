// A small choice list: a button that shows the choice and opens the options under it.
// Used instead of <select>: WebKit restyles a native select from script, which the
// panel's style-src policy refuses.
import { ChevronDown } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

export function Picker<T extends string>({
  id,
  value,
  options,
  label,
  labelOf,
  disabled,
  testId,
  onChange,
}: {
  id?: string;
  value: T;
  options: readonly T[];
  label?: string;
  labelOf: (v: T) => string;
  disabled: boolean;
  testId: string;
  onChange: (v: T) => void;
}) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: PointerEvent): void => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [open]);
  return (
    <div className="picker" ref={ref}>
      <button
        id={id}
        type="button"
        className="input picker-btn"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        data-testid={testId}
        data-value={value}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{labelOf(value)}</span>
        <ChevronDown aria-hidden="true" size={14} />
      </button>
      {open ? (
        <div className="picker-list" role="listbox" id={listId} aria-label={label}>
          {options.map((o) => (
            <button
              key={o}
              type="button"
              role="option"
              aria-selected={o === value}
              data-testid={`${testId}-${o}`}
              onClick={() => {
                onChange(o);
                setOpen(false);
              }}
            >
              {labelOf(o)}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
