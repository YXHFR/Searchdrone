import { type ReactNode, useCallback, useEffect, useId, useRef, useState } from "react";

/** How long a decision has to be held. */
export const HOLD_MS = 800;

interface Props {
  onDone(): void;
  disabled?: boolean;
  children: ReactNode;
  /** Shown under the label and read out with it, e.g. "håll in 0,8 s". */
  hint: string;
  testId?: string;
}

/**
 * A decision that fires only after it has been held, so a habitual tap approves nothing. The
 * bar fills in CSS over the same time; letting go early cancels. Pointer, Enter and Space.
 */
export function HoldButton({ onDone, disabled = false, children, hint, testId }: Props) {
  const hintId = useId();
  const [holding, setHolding] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const done = useRef(onDone);
  done.current = onDone;

  const release = useCallback((): void => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    setHolding(false);
  }, []);
  const press = (): void => {
    if (disabled || timer.current !== null) return;
    setHolding(true);
    timer.current = setTimeout(() => {
      timer.current = null;
      setHolding(false);
      done.current();
    }, HOLD_MS);
  };
  useEffect(() => release, [release]);
  useEffect(() => {
    if (disabled) release();
  }, [disabled, release]);

  return (
    <button
      type="button"
      className="btn btn-hold"
      data-testid={testId}
      data-holding={holding}
      disabled={disabled}
      aria-describedby={hintId}
      style={{ "--hold-ms": `${HOLD_MS}ms` } as React.CSSProperties}
      onPointerDown={(e) => {
        if (e.button === 0) press();
      }}
      onPointerUp={release}
      onPointerLeave={release}
      onPointerCancel={release}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !e.repeat) {
          e.preventDefault();
          press();
        }
      }}
      onKeyUp={(e) => {
        if (e.key === "Enter" || e.key === " ") release();
      }}
      onBlur={release}
      onContextMenu={(e) => e.preventDefault()}
    >
      <span className="hold-fill" aria-hidden="true" />
      <span className="hold-label">{children}</span>
      <span className="hold-hint" id={hintId}>
        {hint}
      </span>
    </button>
  );
}
