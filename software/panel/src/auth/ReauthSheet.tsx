import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Dialog } from "../ui/Dialog";
import { BOOT_TIMING, pollSession } from "./bootstrap";
import { PinForm } from "./PinForm";

/** 401 in the middle of a flight: the shell stays, this sheet opens above it and below the safety bar. */
export function ReauthSheet() {
  const { t } = useTranslation();
  useEffect(() => {
    const id = setInterval(() => void pollSession(), BOOT_TIMING.reauthPollMs);
    return () => clearInterval(id);
  }, []);
  return (
    <Dialog title={t("login.reauthTitle")} testId="reauth-sheet">
      <PinForm />
    </Dialog>
  );
}
