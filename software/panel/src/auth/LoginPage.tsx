import { useTranslation } from "react-i18next";
import { PinForm } from "./PinForm";

export function LoginPage() {
  const { t } = useTranslation();
  return (
    <main className="page-center" id="main">
      <div className="card login-card">
        <h1>{t("login.title")}</h1>
        <PinForm />
      </div>
    </main>
  );
}
