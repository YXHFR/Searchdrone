// Adapter: the server's Approval (kind charter | command | resume) → the panel's card type.
// A resume approval never becomes a card: only the Resume button releases a hold.
import type { Approval, Charter, Command } from "./types";

export type ApprovalKind = "charter" | "charter_change" | "skill";

export interface PanelApproval {
  approval_id: string;
  kind: ApprovalKind;
  status: "pending" | "approved" | "rejected" | "expired" | "superseded" | "gate_rejected";
  subject_id: string;
  subject_version: number;
  created_at: string;
  summary_sv: string;
  rationale?: string | undefined;
  requested_by: string;
  decided_by: string | null;
  decided_at: string | null;
  expires_at?: string | undefined;
  required_confirmations: string[];
  gate_reason_codes?: string[] | undefined;
  default_autonomy_mode: Approval["default_autonomy_mode"];
  charter?: Charter | undefined;
  base_charter?: Charter | undefined;
  command?: Command | undefined;
  subject_loading: boolean;
}

export interface Subjects {
  charters: Record<string, Charter>;
  commands: Record<string, Command>;
}

/** The approved charter a change is measured against (the newest approved one). */
function approvedCharter(s: Subjects, except: string): Charter | undefined {
  return Object.values(s.charters)
    .filter((c) => c.status === "approved" && c.id !== except)
    .sort((a, b) => b.version - a.version)[0];
}

export function toPanelApproval(a: Approval, s: Subjects): PanelApproval | null {
  if (a.kind === "resume") return null;
  const base = {
    approval_id: a.id,
    subject_id: a.subject_id,
    subject_version: a.subject_version,
    created_at: a.requested_at,
    summary_sv: a.summary_sv,
    requested_by: a.requested_by,
    decided_by: a.decided_by,
    decided_at: a.decided_at,
    expires_at: a.expires_at ?? undefined,
    required_confirmations: [...a.required_confirmations],
    gate_reason_codes: a.gate?.reason_codes,
    default_autonomy_mode: a.default_autonomy_mode,
  };
  if (a.kind === "charter") {
    const charter = s.charters[a.subject_id];
    const prior = approvedCharter(s, a.subject_id);
    // A change only while this one is still a proposal next to an approved charter.
    const isChange = prior !== undefined && (charter?.status ?? "proposed") === "proposed";
    return {
      ...base,
      kind: isChange ? "charter_change" : "charter",
      status: a.status,
      rationale: charter?.body.notes ?? undefined,
      charter,
      base_charter: isChange ? prior : undefined,
      subject_loading: charter === undefined,
    };
  }
  const command = s.commands[a.subject_id];
  const gateRejected = a.status === "rejected" && command?.status === "rejected_by_gate";
  return {
    ...base,
    kind: "skill",
    status: gateRejected ? "gate_rejected" : a.status,
    rationale: command?.rationale_sv ?? undefined,
    command,
    gate_reason_codes:
      gateRejected && command?.status_reason ? [command.status_reason] : base.gate_reason_codes,
    subject_loading: command === undefined,
  };
}

/** The mode preselected on a charter card: the server's choice, else A. Never computed here. */
export function defaultAutonomy(a: PanelApproval): "APPROVE_EACH" | "WITHIN_CHARTER" {
  return a.default_autonomy_mode ?? "APPROVE_EACH";
}
