import { type ReactNode, useEffect, useId, useRef } from "react";

interface Props {
  title: string;
  children: ReactNode;
  onClose?: () => void;
  testId?: string;
}

/**
 * The only dialog component. It ends above the safety bar and never covers it.
 * window.confirm/alert/prompt are never used (they would cover the whole page).
 */
export function Dialog({ title, children, onClose, testId }: Props) {
  const titleId = useId();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.focus();
    if (!onClose) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="dialog-backdrop">
      <div
        ref={ref}
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        data-testid={testId}
      >
        <h2 id={titleId}>{title}</h2>
        {children}
      </div>
    </div>
  );
}
