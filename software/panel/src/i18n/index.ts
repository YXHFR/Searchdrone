import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import sv from "./sv.json";

void i18next.use(initReactI18next).init({
  lng: "sv",
  fallbackLng: "sv",
  resources: { sv: { translation: sv } },
  interpolation: { escapeValue: false }, // React escapes; text is never injected as HTML
  returnNull: false,
});

export default i18next;
