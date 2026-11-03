// A small choice list: a button that shows the choice and opens the options under it.
// Used instead of <select>: WebKit restyles a native select from script, which the
// panel's style-src policy refuses.
// Keyboard (as a native select): ↓/↑ on the button opens the list on the current choice;
// in the list ↓/↑ move, Home/End jump, Enter/Space choose, Escape and Tab close it.
import { ChevronDown } from "lucide-react";
import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";

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
  // The option that holds the keyboard focus while the list is open; null: focus stays
  // on the button (the list was opened with a pointer).
  const [active, setActive] = useState<number | null>(null);
  const listId = useId();
  const ref = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const close = (refocus: boolean): void => {
    setOpen(false);
    setActive(null);
    if (refocus) buttonRef.current?.focus();
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: globalThis.KeyboardEvent): void => {
      if (e.key !== "Escape") return;
      const inside = ref.current?.contains(document.activeElement) ?? false;
      setOpen(false);
      setActive(null);
      if (inside) buttonRef.current?.focus();
    };
    const onDown = (e: PointerEvent): void => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setActive(null);
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  useEffect(() => {
    if (open && active !== null) optionRefs.current[active]?.focus();
  }, [open, active]);

  const current = Math.max(0, options.indexOf(value));
  const last = options.length - 1;

  const onButtonKey = (e: KeyboardEvent<HTMLButtonElement>): void => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    if (options.length === 0) return;
    setOpen(true);
    setActive(active ?? current);
  };

  const onListKey = (e: KeyboardEvent<HTMLDivElement>): void => {
    const at = active ?? current;
    const next: Record<string, number> = {
      ArrowDown: Math.min(last, at + 1),
      ArrowUp: Math.max(0, at - 1),
      Home: 0,
      End: last,
    };
    if (e.key in next) {
      e.preventDefault();
      setActive(next[e.key] ?? at);
    } else if (e.key === "Tab") {
      close(false);
    }
  };

  return (
    <div className="picker" ref={ref}>
      <button
        ref={buttonRef}
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
        onClick={() => (open ? close(false) : setOpen(true))}
        onKeyDown={onButtonKey}
      >
        <span>{labelOf(value)}</span>
        <ChevronDown aria-hidden="true" size={14} />
      </button>
      {open ? (
        <div
          className="picker-list"
          role="listbox"
          id={listId}
          aria-label={label}
          tabIndex={-1}
          onKeyDown={onListKey}
        >
          {options.map((o, i) => (
            <button
              key={o}
              ref={(el) => {
                optionRefs.current[i] = el;
              }}
              type="button"
              role="option"
              aria-selected={o === value}
              tabIndex={i === (active ?? current) ? 0 : -1}
              data-testid={`${testId}-${o}`}
              onClick={() => {
                onChange(o);
                close(active !== null);
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
