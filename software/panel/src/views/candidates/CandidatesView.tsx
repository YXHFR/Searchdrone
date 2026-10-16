import { useTranslation } from "react-i18next";
import type { ViewProps } from "../registry";

export default function CandidatesView(_: ViewProps) {
  const { t } = useTranslation();
  return (
    <div className="view-body view-empty">
      <p>{t("views.candidates.stub")}</p>
      <p className="muted">{t("views.candidates.onlyYou")}</p>
    </div>
  );
}
