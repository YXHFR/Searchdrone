import { type FormEvent, useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { loginWithPin } from "./bootstrap";

export function PinForm() {
  const { t } = useTranslation();
  const id = useId();
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const valid = /^[0-9]{6}$/.test(pin);

  async function submit(e: FormEvent): Promise<void> {
    e.preventDefault();
    if (!valid || busy) return;
    setBusy(true);
    const err = await loginWithPin(pin);
    setPin("");
    setBusy(false);
    setError(err);
  }

  return (
    <form className="pin-form" onSubmit={(e) => void submit(e)}>
      <p className="muted">{t("login.help")}</p>
      <label htmlFor={id}>{t("login.pin")}</label>
      <input
        id={id}
        className="input pin-input"
        inputMode="numeric"
        pattern="[0-9]{6}"
        maxLength={6}
        autoComplete="one-time-code"
        value={pin}
        onChange={(e) => setPin(e.target.value.replace(/\D/g, "").slice(0, 6))}
      />
      <button type="submit" className="btn btn-primary" disabled={!valid || busy}>
        {t("login.submit")}
      </button>
      {error ? (
        <p className="error" role="alert">
          {t(error)}
        </p>
      ) : null}
    </form>
  );
}
