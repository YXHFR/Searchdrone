import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HOLD_MS, HoldButton } from "./HoldButton";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

function setup(disabled = false) {
  const onDone = vi.fn();
  render(
    <HoldButton onDone={onDone} hint="håll in" disabled={disabled}>
      Godkänn
    </HoldButton>,
  );
  return { onDone, button: screen.getByRole("button", { name: /Godkänn/ }) };
}

describe("HoldButton", () => {
  it("does nothing on a click", () => {
    const { onDone, button } = setup();
    fireEvent.pointerDown(button, { button: 0 });
    fireEvent.pointerUp(button);
    fireEvent.click(button);
    act(() => vi.advanceTimersByTime(HOLD_MS * 2));
    expect(onDone).not.toHaveBeenCalled();
  });

  it("fires once after being held for the full time", () => {
    const { onDone, button } = setup();
    fireEvent.pointerDown(button, { button: 0 });
    act(() => vi.advanceTimersByTime(HOLD_MS - 1));
    expect(onDone).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1));
    expect(onDone).toHaveBeenCalledTimes(1);
    fireEvent.pointerUp(button);
    act(() => vi.advanceTimersByTime(HOLD_MS));
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("cancels when the pointer leaves early", () => {
    const { onDone, button } = setup();
    fireEvent.pointerDown(button, { button: 0 });
    act(() => vi.advanceTimersByTime(HOLD_MS / 2));
    fireEvent.pointerLeave(button);
    act(() => vi.advanceTimersByTime(HOLD_MS));
    expect(onDone).not.toHaveBeenCalled();
  });

  it("works from the keyboard, ignoring key repeat", () => {
    const { onDone, button } = setup();
    fireEvent.keyDown(button, { key: "Enter" });
    fireEvent.keyDown(button, { key: "Enter", repeat: true });
    act(() => vi.advanceTimersByTime(HOLD_MS));
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("cannot be held while disabled", () => {
    const { onDone, button } = setup(true);
    fireEvent.pointerDown(button, { button: 0 });
    act(() => vi.advanceTimersByTime(HOLD_MS));
    expect(onDone).not.toHaveBeenCalled();
  });
});
