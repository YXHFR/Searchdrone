import { ChevronDown } from "lucide-react";
import { type ReactNode, useEffect, useId, useRef, useState } from "react";

interface Props {
  label: string;
  icon: ReactNode;
  testId?: string;
  /** A labelled map tool (icon, word and chevron) instead of a bare icon button. */
  tool?: boolean;
  active?: boolean;
  children: (close: () => void) => ReactNode;
}

/** Small drop-down: button + list. Closes on Escape and outside click. */
export function Menu({ label, icon, testId, tool = false, active, children }: Props) {
  const [open, setOpen] = useState(false);
  const id = useId();
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
    <div className="menu" ref={ref}>
      <button
        type="button"
        className={tool ? "tool" : "icon-btn"}
        aria-expanded={open}
        aria-controls={id}
        aria-label={tool ? undefined : label}
        aria-pressed={active}
        data-testid={testId}
        onClick={() => setOpen((o) => !o)}
      >
        {icon}
        {tool ? (
          <>
            {label}
            <ChevronDown aria-hidden="true" size={14} className="tool-chev" />
          </>
        ) : null}
      </button>
      {open ? (
        <div className={tool ? "menu-list tool-pop" : "menu-list"} id={id}>
          {children(() => setOpen(false))}
        </div>
      ) : null}
    </div>
  );
}
