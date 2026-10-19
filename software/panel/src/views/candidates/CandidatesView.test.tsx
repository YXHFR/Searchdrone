import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CandidatesView from "./CandidatesView";

describe("CandidatesView (stage 2 stub)", () => {
  it("says plainly that the candidate queue is not built yet and offers no actions", () => {
    render(<CandidatesView vehicleId="sim-01" isCompact={false} />);
    expect(screen.getByText(/kommer i etapp 2/)).toBeTruthy();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.queryAllByRole("listitem")).toHaveLength(0);
  });
});
