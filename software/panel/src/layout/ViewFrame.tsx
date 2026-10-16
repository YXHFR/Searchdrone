import { Maximize2, Minimize2, X } from "lucide-react";
import { Suspense, useId } from "react";
import { useTranslation } from "react-i18next";
import { ErrorBoundary } from "../app/ErrorBoundary";
import { shouldCrash } from "../lib/testHooks";
import { useLayoutStore } from "../state/layoutStore";
import { VIEW_REGISTRY, type ViewProps } from "../views/registry";
import { canSplitAt, type Path, VIEW_IDS, type ViewId, visibleViews } from "./layoutModel";

function Crash(): never {
  throw new Error("test crash");
}

export function ViewContent({ view, ...props }: ViewProps & { view: ViewId }) {
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

interface Props {
  view: ViewId;
  path: Path;
  vehicleId: string;
}

/** A pane: title bar with full screen, "show here", split and close, then the view. */
export function ViewFrame({ view, path, vehicleId }: Props) {
  const { t } = useTranslation();
  const titleId = useId();
  const layout = useLayoutStore((s) => s.layout);
  const vp = useLayoutStore((s) => s.vp);
  const store = useLayoutStore.getState;
  const maximized = layout.maximized === view;
  const shown = visibleViews({ ...layout, maximized: null });
  const others = VIEW_IDS.filter((v) => v !== view);
  const canSplit = canSplitAt(layout, path, vp);
  const Icon = VIEW_REGISTRY[view].icon;
  const title = t(VIEW_REGISTRY[view].titleKey);

  return (
    <section
      className="frame"
      aria-labelledby={titleId}
      data-view={view}
      data-testid={`view-${view}`}
    >
      <header className="frame-head">
        <h2 id={titleId} className="frame-title">
          <Icon aria-hidden="true" size={16} />
          {title}
        </h2>
        <div className="frame-tools">
          {!maximized ? (
            <select
              className="select"
              aria-label={t("layout.showHere", { view: title })}
              value=""
              onChange={(e) => store().replace(path, e.target.value as ViewId)}
            >
              <option value="" disabled>
                {t("layout.showHereShort")}
              </option>
              {others.map((v) => (
                <option key={v} value={v}>
                  {t(VIEW_REGISTRY[v].titleKey)}
                </option>
              ))}
            </select>
          ) : null}
          {!maximized && canSplit ? (
            <select
              className="select"
              aria-label={t("layout.split", { view: title })}
              value=""
              onChange={(e) => {
                const [dir, v] = e.target.value.split(":") as ["row" | "col", ViewId];
                store().split(path, dir, v);
              }}
            >
              <option value="" disabled>
                {t("layout.splitShort")}
              </option>
              {others
                .filter((v) => !shown.includes(v))
                .flatMap((v) => [
                  <option key={`row:${v}`} value={`row:${v}`}>
                    {t("layout.splitRight", { view: t(VIEW_REGISTRY[v].titleKey) })}
                  </option>,
                  <option key={`col:${v}`} value={`col:${v}`}>
                    {t("layout.splitBelow", { view: t(VIEW_REGISTRY[v].titleKey) })}
                  </option>,
                ])}
            </select>
          ) : null}
          <button
            type="button"
            className="icon-btn"
            aria-label={
              maximized
                ? t("layout.restore", { view: title })
                : t("layout.maximize", { view: title })
            }
            aria-pressed={maximized}
            data-testid={`maximize-${view}`}
            onClick={() => (maximized ? store().restore() : store().maximize(view))}
          >
            {maximized ? (
              <Minimize2 aria-hidden="true" size={16} />
            ) : (
              <Maximize2 aria-hidden="true" size={16} />
            )}
          </button>
          {!maximized && shown.length > 1 ? (
            <button
              type="button"
              className="icon-btn"
              aria-label={t("layout.close", { view: title })}
              onClick={() => store().close(path)}
            >
              <X aria-hidden="true" size={16} />
            </button>
          ) : null}
        </div>
      </header>
      <div className="frame-body">
        <ViewContent view={view} vehicleId={vehicleId} isMaximized={maximized} isCompact={false} />
      </div>
    </section>
  );
}
