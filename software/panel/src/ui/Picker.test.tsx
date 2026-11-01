import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Picker } from "./Picker";

const OPTIONS = ["a", "b", "c", "d"] as const;
type Opt = (typeof OPTIONS)[number];

function setup(start: Opt = "b") {
  const onChange = vi.fn();
  function Host() {
    const [v, setV] = useState<Opt>(start);
    return (
      <Picker
        value={v}
        options={OPTIONS}
        label="Val"
        labelOf={(o) => o.toUpperCase()}
        disabled={false}
        testId="p"
        onChange={(o) => {
          onChange(o);
          setV(o);
        }}
      />
    );
  }
  render(<Host />);
  return { onChange, button: screen.getByTestId("p") };
}

const focused = () => document.activeElement?.getAttribute("data-testid");

describe("Picker keyboard", () => {
  it("↓ on the button opens the list with the focus on the current choice", () => {
    const { button } = setup("b");
    button.focus();
    fireEvent.keyDown(button, { key: "ArrowDown" });
    expect(button.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("listbox")).toBeTruthy();
    expect(focused()).toBe("p-b");
  });

  it("↓/↑ move, stop at the ends, Home/End jump; only the focused option is tabbable", () => {
    const { button } = setup("b");
    fireEvent.keyDown(button, { key: "ArrowUp" });
    const list = screen.getByRole("listbox");
    fireEvent.keyDown(list, { key: "ArrowDown" });
    expect(focused()).toBe("p-c");
    fireEvent.keyDown(list, { key: "ArrowDown" });
    fireEvent.keyDown(list, { key: "ArrowDown" });
    expect(focused()).toBe("p-d");
    fireEvent.keyDown(list, { key: "Home" });
    expect(focused()).toBe("p-a");
    fireEvent.keyDown(list, { key: "ArrowUp" });
    expect(focused()).toBe("p-a");
    fireEvent.keyDown(list, { key: "End" });
    expect(focused()).toBe("p-d");
    const tabbable = screen.getAllByRole("option").filter((o) => o.tabIndex === 0);
    expect(tabbable.map((o) => o.dataset.testid)).toEqual(["p-d"]);
  });

  it("choosing with the keyboard sets the value, closes and returns the focus", () => {
    const { button, onChange } = setup("a");
    fireEvent.keyDown(button, { key: "ArrowDown" });
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "ArrowDown" });
    // Enter/Space on the focused option button is its native click.
    fireEvent.click(screen.getByTestId("p-b"));
    expect(onChange).toHaveBeenCalledWith("b");
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(button.textContent).toBe("B");
    expect(document.activeElement).toBe(button);
  });

  it("Escape closes without a change and returns the focus to the button", () => {
    const { button, onChange } = setup("c");
    fireEvent.keyDown(button, { key: "ArrowDown" });
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "ArrowDown" });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(button);
  });

  it("Tab closes the list", () => {
    const { button } = setup();
    fireEvent.keyDown(button, { key: "ArrowDown" });
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "Tab" });
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(button.getAttribute("aria-expanded")).toBe("false");
  });

  it("a pointer click still opens the list and leaves the focus on the button", () => {
    const { button, onChange } = setup("a");
    button.focus();
    fireEvent.click(button);
    expect(screen.getByRole("listbox")).toBeTruthy();
    expect(document.activeElement).toBe(button);
    fireEvent.click(screen.getByTestId("p-c"));
    expect(onChange).toHaveBeenCalledWith("c");
    expect(screen.queryByRole("listbox")).toBeNull();
  });
});
