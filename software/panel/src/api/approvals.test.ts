import { describe, expect, it } from "vitest";
import { defaultAutonomy, toPanelApproval } from "./approvals";
import type { Approval, Charter, Command } from "./types";

const base: Approval = {
  id: "AP1",
  kind: "charter",
  mission_id: "M1",
  vehicle_id: "sim-01",
  subject_id: "C2",
  subject_version: 2,
  status: "pending",
  summary_sv: "Ny ram",
  requested_at: "2026-10-16T10:00:00Z",
  requested_by: "brain:commander",
  required_confirmations: [],
  default_autonomy_mode: null,
  decided_at: null,
  decided_by: null,
  expires_at: null,
  gate: null,
  reason: null,
};
const charter = (id: string, status: Charter["status"], version: number) =>
  ({ id, status, version, body: { notes: `n${version}` } }) as unknown as Charter;
const command = (status: Command["status"], reason: string | null = null) =>
  ({ id: "K1", status, status_reason: reason, rationale_sv: "för att" }) as unknown as Command;

describe("toPanelApproval", () => {
  it("a charter with no approved one is a new charter; with one it is a change", () => {
    const fresh = toPanelApproval(base, {
      charters: { C2: charter("C2", "proposed", 2) },
      commands: {},
    });
    expect(fresh?.kind).toBe("charter");
    const change = toPanelApproval(base, {
      charters: { C1: charter("C1", "approved", 1), C2: charter("C2", "proposed", 2) },
      commands: {},
    });
    expect(change?.kind).toBe("charter_change");
    expect(change?.base_charter?.id).toBe("C1");
  });

  it("maps the fields: created_at = requested_at, summary, rationale from the charter notes", () => {
    const a = toPanelApproval(base, {
      charters: { C2: charter("C2", "proposed", 2) },
      commands: {},
    });
    expect(a).toMatchObject({
      approval_id: "AP1",
      created_at: "2026-10-16T10:00:00Z",
      summary_sv: "Ny ram",
      rationale: "n2",
      subject_version: 2,
      subject_loading: false,
    });
  });

  it("a missing subject keeps the card loading", () => {
    expect(toPanelApproval(base, { charters: {}, commands: {} })?.subject_loading).toBe(true);
  });

  it("an order card; rejected by the gate becomes gate_rejected with the reason", () => {
    const a = { ...base, kind: "command" as const, subject_id: "K1" };
    expect(
      toPanelApproval(a, { charters: {}, commands: { K1: command("pending_approval") } })?.kind,
    ).toBe("skill");
    const r = toPanelApproval(
      { ...a, status: "rejected" },
      { charters: {}, commands: { K1: command("rejected_by_gate", "OUTSIDE_AREA") } },
    );
    expect(r?.status).toBe("gate_rejected");
    expect(r?.gate_reason_codes).toEqual(["OUTSIDE_AREA"]);
  });

  it("a resume approval never becomes a card", () => {
    expect(toPanelApproval({ ...base, kind: "resume" }, { charters: {}, commands: {} })).toBeNull();
  });

  it("the preselected mode is the server's, else A; never computed in the panel", () => {
    const mode = (m: Approval["default_autonomy_mode"]) => {
      const card = toPanelApproval(
        { ...base, default_autonomy_mode: m },
        { charters: {}, commands: {} },
      );
      return card ? defaultAutonomy(card) : null;
    };
    expect(mode("WITHIN_CHARTER")).toBe("WITHIN_CHARTER");
    expect(mode("APPROVE_EACH")).toBe("APPROVE_EACH");
    expect(mode(null)).toBe("APPROVE_EACH");
  });
});
