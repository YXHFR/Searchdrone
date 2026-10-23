import { useTranslation } from "react-i18next";
import { type ProbMeta, rowPercent } from "./probability";

/** The probability layers' legend, the rest-of-world share and what the map is built on. */
export function ProbabilityLegend({
  meta,
  poa,
  pod,
}: {
  meta: ProbMeta;
  poa: boolean;
  pod: boolean;
}) {
  const { t } = useTranslation();
  return (
    <div className="map-prob" data-testid="prob-legend">
      {poa ? (
        <p className="map-prob-row">
          <span className="lbl">{t("views.map.layer.probability")}</span>
          <span className="map-prob-scale map-prob-poa" aria-hidden="true" />
          <span className="muted">{t("views.map.prob.scale")}</span>
        </p>
      ) : null}
      {pod ? (
        <p className="map-prob-row">
          <span className="lbl">{t("views.map.layer.pod")}</span>
          <span className="map-prob-scale map-prob-pod" aria-hidden="true" />
          <span className="muted">{t("views.map.prob.podScale")}</span>
        </p>
      ) : null}
      <p data-testid="prob-row">{t("views.map.prob.row", { pct: rowPercent(meta.row) })}</p>
      {meta.frozen ? <p data-testid="prob-frozen">{t("views.map.prob.frozen")}</p> : null}
      {meta.notes.map((n) => (
        <p key={n} className="muted" data-testid={`prob-note-${n}`}>
          {t(`views.map.prob.note.${n}`)}
        </p>
      ))}
    </div>
  );
}
