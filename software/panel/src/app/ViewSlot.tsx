import { Suspense } from "react";
import { useTranslation } from "react-i18next";
import { shouldCrash } from "../lib/testHooks";
import { VIEW_REGISTRY, type ViewId, type ViewProps } from "../views/registry";
import { ErrorBoundary } from "./ErrorBoundary";

function Crash(): never {
  throw new Error("test crash");
}

/** One view behind its own error boundary: a view that fails never takes the panel along. */
export function ViewSlot({ view, ...props }: ViewProps & { view: ViewId }) {
  const { t } = useTranslation();
  const Comp = VIEW_REGISTRY[view].component;
  return (
    <ErrorBoundary
      fallback={(reset) => (
        <div className="view-body view-empty" role="alert">
          <p>{t("views.crashed")}</p>
          <button type="button" className="btn" onClick={reset}>
            {t("views.reload")}
          </button>
        </div>
      )}
    >
      <Suspense fallback={<p className="view-body muted">{t("common.loading")}</p>}>
        {shouldCrash(view) ? <Crash /> : <Comp {...props} />}
      </Suspense>
    </ErrorBoundary>
  );
}
