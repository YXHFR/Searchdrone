import { useTranslation } from "react-i18next";
import type { PanelApproval } from "../../../api/approvals";
import type { MissionCharter } from "../../../api/types";
import { areaM2, centroid, distanceM, extentM } from "../../../lib/geo";
import { useLiveStore } from "../../../state/liveStore";
import { useSessionStore } from "../../../state/sessionStore";
import {
  changedFromDefault,
  charterDiff,
  charterProblems,
  type DiffRow,
  type Problem,
} from "./limits";

const nf = (digits: number) =>
  new Intl.NumberFormat("sv-SE", { maximumFractionDigits: digits, minimumFractionDigits: 0 });

const SHOWN = new Set([
  "area",
  "sectors",
  "alt_min_agl_m",
  "alt_max_agl_m",
  "default_alt_agl_m",
  "max_speed_ms",
  "max_range_from_pilot_m",
  "pilot_position",
  "battery_rtl_pct",
  "geofence_buffer_m",
  "home",
  "on_ai_silent",
  "ai_silent_timeout_s",
  "allowed_skills",
  "no_personal_data",
  "notes",
  // server fields, not shown as "other"
  "charter_id",
  "version",
  "approved_at",
  "approved_by",
  "fence_hash",
  "mission_id",
  "vehicle_id",
  "autonomy_mode",
]);

function compass(dx: number, dy: number): string {
  const dirs = [
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
  ];
  const a = (Math.atan2(dx, dy) * 180) / Math.PI;
  return dirs[Math.round(((a + 360) % 360) / 45) % 8] ?? "north";
}

export function useCharterProblems(c: MissionCharter | undefined): Problem[] {
  const abs = useSessionStore((s) => s.config?.limits_absolute ?? null);
  const telHome = useLiveStore((s) => s.telemetry?.home ?? null);
  if (!c || !abs) return [];
  return charterProblems(c, abs, telHome);
}

export function ProblemList({ problems }: { problems: Problem[] }) {
  const { t } = useTranslation();
  if (problems.length === 0) return null;
  return (
    <ul className="card-problems" data-testid="card-problems">
      {problems.map((p) => (
        <li key={`${p.key}-${p.field}`} className="error">
          {t("limits.cannotApprove")}: {t(p.key, { ...p, field: t(`charter.field.${p.field}`) })}
        </li>
      ))}
    </ul>
  );
}

function Row({
  label,
  children,
  bad,
}: {
  label: string;
  children: React.ReactNode;
  bad?: boolean;
}) {
  return (
    <div className="kv" data-bad={bad ? "true" : undefined}>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

/** All fields of the charter (unknown ones raw), with "changed from standard" marks. */
export function CharterFields({ c }: { c: MissionCharter }) {
  const { t } = useTranslation();
  const defaults = useSessionStore((s) => s.config?.limits_default ?? {});
  const problems = useCharterProblems(c);
  const bad = new Set(problems.map((p) => p.field));
  const mark = (
    field: "max_alt_agl_m" | "max_speed_ms" | "max_range_from_pilot_m" | "battery_rtl_pct",
    value: number | undefined,
  ) => {
    const ch = changedFromDefault(field, value, defaults);
    return ch ? ` · ${t(`charter.changed.${ch}`)}` : "";
  };
  const m2 = areaM2(c.area);
  const ext = extentM(c.area);
  const mid = centroid(c.area);
  const homeLL = { lat: c.home.lat, lon: c.home.lon };
  const midText = mid
    ? t("charter.areaMid", {
        m: nf(0).format(distanceM(homeLL, mid)),
        dir: t(`charter.dir.${compass(mid.lon - homeLL.lon, mid.lat - homeLL.lat)}`),
      })
    : "";
  const pilotM = distanceM(c.pilot_position, homeLL);
  const silent = c.on_ai_silent ?? "continue_plan";
  const extra = Object.entries(c as unknown as Record<string, unknown>).filter(
    ([k]) => !SHOWN.has(k),
  );
  return (
    <dl className="charter-fields">
      <Row label={t("charter.field.area")} bad={bad.has("area")}>
        {t("charter.areaValue", {
          ha: nf(1).format(m2 / 10_000),
          w: ext ? nf(0).format(ext.w) : "–",
          h: ext ? nf(0).format(ext.h) : "–",
        })}
        {midText ? `, ${midText}` : ""}
      </Row>
      <Row label={t("charter.field.sectors")}>
        {t("charter.sectorsValue", {
          n: c.sectors.length,
          names: c.sectors.map((s) => s.name).join(", "),
          alt: c.default_alt_agl_m ?? "–",
        })}
      </Row>
      <Row
        label={t("charter.field.alt_max_agl_m")}
        bad={bad.has("alt_max_agl_m") || bad.has("alt_min_agl_m")}
      >
        {t("charter.altValue", { min: c.alt_min_agl_m ?? "–", max: c.alt_max_agl_m ?? "–" })}
        {mark("max_alt_agl_m", c.alt_max_agl_m)}
      </Row>
      <Row label={t("charter.field.max_speed_ms")} bad={bad.has("max_speed_ms")}>
        {t("charter.speedValue", { v: c.max_speed_ms ?? "–" })}
        {mark("max_speed_ms", c.max_speed_ms)}
      </Row>
      <Row
        label={t("charter.field.max_range_from_pilot_m")}
        bad={bad.has("max_range_from_pilot_m") || bad.has("pilot_position")}
      >
        {t("charter.rangeValue", { v: c.max_range_from_pilot_m ?? "–" })}
        {mark("max_range_from_pilot_m", c.max_range_from_pilot_m)}
        {" · "}
        {pilotM <= 10
          ? t("charter.pilotAtHome")
          : t("charter.pilotFromHome", { m: nf(0).format(pilotM) })}
      </Row>
      <Row label={t("charter.field.battery_rtl_pct")} bad={bad.has("battery_rtl_pct")}>
        {t("charter.batteryValue", { v: c.battery_rtl_pct ?? "–" })}
        {mark("battery_rtl_pct", c.battery_rtl_pct)}
      </Row>
      <Row label={t("charter.field.geofence_buffer_m")}>
        {t("charter.bufferValue", { v: c.geofence_buffer_m ?? "–" })}
      </Row>
      <Row label={t("charter.field.home")} bad={bad.has("home")}>
        {`${nf(5).format(c.home.lat)}, ${nf(5).format(c.home.lon)}`}
      </Row>
      <Row label={t("charter.field.on_ai_silent")}>
        {t(`charter.silent.${silent}`, { s: c.ai_silent_timeout_s ?? "–" })}
      </Row>
      <Row label={t("charter.field.allowed_skills")}>
        <span className="chips">
          {c.allowed_skills.map((s) => (
            <span key={s} className="chip">
              {t(`skills.${s}`, { defaultValue: s })}
            </span>
          ))}
        </span>
      </Row>
      {"no_personal_data" in c ? (
        <Row label={t("charter.field.no_personal_data")}>
          {t(
            (c as { no_personal_data?: boolean }).no_personal_data
              ? "charter.noPersonal"
              : "charter.maybePersonal",
          )}
        </Row>
      ) : null}
      {c.notes ? <Row label={t("charter.field.notes")}>{c.notes}</Row> : null}
      {extra.map(([k, v]) => (
        <Row key={k} label={k}>
          {JSON.stringify(v)}
        </Row>
      ))}
    </dl>
  );
}

function show(v: unknown): string {
  if (v === undefined) return "–";
  if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") return String(v);
  if (Array.isArray(v) && v.every((x) => typeof x === "string")) return v.join(", ");
  if (v && typeof v === "object" && "lat" in v && "lon" in v) {
    const p = v as { lat: number; lon: number };
    return `${nf(5).format(p.lat)}, ${nf(5).format(p.lon)}`;
  }
  if (v && typeof v === "object" && "type" in v) return `${nf(1).format(areaM2(v) / 10_000)} ha`;
  return JSON.stringify(v);
}

/** Generic diff against the approved charter: every changed field, unknown ones too. */
export function CharterDiff({ before, after }: { before: MissionCharter; after: MissionCharter }) {
  const { t } = useTranslation();
  const rows: DiffRow[] = charterDiff(before, after);
  if (rows.length === 0) return <p className="muted">{t("charter.noChange")}</p>;
  return (
    <ul className="charter-diff" data-testid="charter-diff">
      {rows.map((r) => (
        <li key={r.field} data-looser={r.looser ? "true" : undefined}>
          {r.known ? t(`charter.field.${r.field}`) : t("charter.unknownField", { name: r.field })}
          {": "}
          {show(r.before)} → {show(r.after)}
          {r.looser ? ` ${t("charter.looser")}` : ""}
          {!r.known ? ` ${t("charter.changedUnknown")}` : ""}
        </li>
      ))}
    </ul>
  );
}

/** The four facts a pilot decides on; every field is one press away below. */
export function CharterKeyFacts({ c }: { c: MissionCharter }) {
  const { t } = useTranslation();
  const mid = centroid(c.area);
  const homeLL = { lat: c.home.lat, lon: c.home.lon };
  const where = mid
    ? t("charter.areaMid", {
        m: nf(0).format(distanceM(homeLL, mid)),
        dir: t(`charter.dir.${compass(mid.lon - homeLL.lon, mid.lat - homeLL.lat)}`),
      })
    : "";
  const dash = (v: number | null | undefined): string => (v == null ? "–" : nf(1).format(v));
  const facts: [string, string][] = [
    ["area", `${nf(1).format(areaM2(c.area) / 10_000)} ha${where ? ` · ${where}` : ""}`],
    ["alt", `${dash(c.alt_min_agl_m)}–${dash(c.alt_max_agl_m)} m`],
    ["speed", `≤ ${dash(c.max_speed_ms)} m/s`],
    ["rtl", t("charter.key.rtlValue", { pct: dash(c.battery_rtl_pct) })],
  ];
  return (
    <dl className="facts" data-testid="charter-facts">
      {facts.map(([k, v]) => (
        <div key={k} data-fact={k}>
          <dt>{t(`charter.key.${k}`)}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function CharterBody({ approval }: { approval: PanelApproval }) {
  const { t } = useTranslation();
  const c = approval.charter?.body;
  if (!c) return <p className="muted">{t("common.loading")}</p>;
  return (
    <>
      {approval.kind === "charter_change" && approval.base_charter ? (
        <CharterDiff before={approval.base_charter.body} after={c} />
      ) : null}
      <CharterKeyFacts c={c} />
      <details className="all-fields">
        <summary>{t("charter.allFields", { v: approval.subject_version })}</summary>
        <CharterFields c={c} />
      </details>
    </>
  );
}
