import { useTranslation } from "react-i18next";
import type { PanelApproval } from "../../../api/approvals";
import { formatLength, lengthM, orderPath } from "../../map/pathData";

const nf = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 5 });

/** An order waiting for a click: the skill in Swedish and its parameters in readable form. */
export function SkillBody({ approval }: { approval: PanelApproval }) {
  const { t } = useTranslation();
  const c = approval.command;
  if (!c) return <p className="muted">{t("common.loading")}</p>;
  const p = c.params as Record<string, unknown>;
  const lat = typeof p.lat === "number" ? p.lat : null;
  const lon = typeof p.lon === "number" ? p.lon : null;
  const alt = typeof p.alt_m === "number" ? p.alt_m : null;
  const path = orderPath(c);
  const hide = ["lat", "lon", "alt_m", "polygon", ...(path ? ["points", "label"] : [])];
  const rest = Object.entries(p).filter(([k]) => !hide.includes(k));
  return (
    <div className="skill-body">
      {path ? (
        <p data-testid="skill-path">
          {t("cards.skill.path", {
            label: typeof p.label === "string" ? p.label : "",
            n: path.length,
            length: formatLength(lengthM(path)),
            alt: alt ?? "–",
          })}
        </p>
      ) : lat !== null && lon !== null ? (
        <p>
          {t("cards.skill.point", {
            lat: nf.format(lat),
            lon: nf.format(lon),
            alt: alt ?? "–",
          })}
        </p>
      ) : alt !== null ? (
        <p>{t("cards.skill.alt", { alt })}</p>
      ) : null}
      {"polygon" in p && p.polygon ? <p>{t("cards.skill.area")}</p> : null}
      {rest.length > 0 ? (
        <dl className="charter-fields">
          {rest.map(([k, v]) => (
            <div className="kv" key={k}>
              <dt>{k}</dt>
              <dd>{typeof v === "string" ? v : JSON.stringify(v)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  );
}
