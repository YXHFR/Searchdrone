// The card of one found object on the map: class, how sure the source is, where, the
// thumbnail and what kind of source it was. A find is a hint, never confirmed here.
// The thumbnail is a background, not an <img>: WebKit styles an image's built-in overlay
// with an inline <style> that the panel's CSP blocks and reports.
import { X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { paths } from "../../api/paths";
import type { Find } from "../../api/types";

const pct = new Intl.NumberFormat("sv-SE", { style: "percent", maximumFractionDigits: 0 });
const deg = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 5 });

export function FindCard({
  find,
  missionId,
  onClose,
}: {
  find: Find;
  missionId: string;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  return (
    <section className="map-find" data-testid="find-card" aria-label={t("views.map.find.title")}>
      <header>
        <strong>{find.name_sv}</strong>
        <button
          type="button"
          className="icon-btn"
          aria-label={t("views.map.find.close")}
          onClick={onClose}
        >
          <X aria-hidden="true" size={16} />
        </button>
      </header>
      {find.thumb_detection_id ? (
        <div
          data-testid="find-thumb"
          className="map-find-thumb"
          role="img"
          aria-label={find.name_sv}
          style={{
            backgroundImage: `url("${paths.detectionCrop(missionId, find.thumb_detection_id)}")`,
          }}
        />
      ) : (
        <p className="muted">{t("views.map.find.noThumb")}</p>
      )}
      <p>{t("views.map.find.score", { score: pct.format(find.score) })}</p>
      <p className="muted">
        {t("views.map.find.where", {
          lat: deg.format(find.lat),
          lon: deg.format(find.lon),
          sigma: Math.round(find.sigma95_m),
        })}
      </p>
      {find.count > 1 ? (
        <p className="muted">{t("views.map.find.seen", { count: find.count })}</p>
      ) : null}
      <p className={find.source === "oracle" ? "warn" : "muted"}>
        {t(`views.map.find.${find.source}`)}
      </p>
      <p className="muted">{t("views.map.find.unconfirmed")}</p>
    </section>
  );
}
