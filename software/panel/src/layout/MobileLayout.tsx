import { PictureInPicture2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useLayoutStore } from "../state/layoutStore";
import { VIEW_REGISTRY } from "../views/registry";
import { VIEW_IDS, type ViewId } from "./layoutModel";
import { ViewContent } from "./ViewFrame";

const CORNERS = ["tl", "tr", "bl", "br"] as const;

/** Phones: one view at a time, a tab row above the safety bar, and an optional picture-in-picture. */
export function MobileLayout({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const layout = useLayoutStore((s) => s.layout);
  const pip = useLayoutStore((s) => s.pip);
  const corner = useLayoutStore((s) => s.pipCorner);
  const store = useLayoutStore.getState;
  const main: ViewId = layout.root.kind === "view" ? layout.root.view : "chat";
  const pipChoice = main === "map" ? "video" : "map";
  const mainTitle = t(VIEW_REGISTRY[main].titleKey);

  return (
    <>
      <main className="mobile-main" id="main" data-testid={`view-${main}`} data-view={main}>
        <div className="mobile-head">
          <h2 className="frame-title">{mainTitle}</h2>
          <button
            type="button"
            className="icon-btn"
            aria-pressed={pip !== null}
            aria-label={
              pip
                ? t("layout.pipOff")
                : t("layout.pipOn", { view: t(VIEW_REGISTRY[pipChoice].titleKey) })
            }
            data-testid="pip-toggle"
            onClick={() => store().setPip(pip ? null : pipChoice)}
          >
            <PictureInPicture2 aria-hidden="true" size={18} />
          </button>
        </div>
        {/* A view taller than the screen must scroll from the keyboard too (axe
            scrollable-region-focusable), even when it has nothing else to focus. */}
        {/* biome-ignore lint/a11y/noNoninteractiveTabindex: a scrolling region needs a tab stop */}
        <section className="frame-body" aria-label={mainTitle} tabIndex={0}>
          <ViewContent view={main} vehicleId={vehicleId} isMaximized={false} isCompact={false} />
        </section>
        {pip ? (
          <div className={`pip pip-${corner}`} data-testid="pip" data-view={pip}>
            <button
              type="button"
              className="pip-swap"
              aria-label={t("layout.pipSwap", { view: t(VIEW_REGISTRY[pip].titleKey) })}
              onClick={() => store().swapPip()}
            />
            <ViewContent view={pip} vehicleId={vehicleId} isMaximized={false} isCompact />
            <div className="pip-corners">
              {CORNERS.filter((c) => c !== corner).map((c) => (
                <button
                  key={c}
                  type="button"
                  className="pip-corner"
                  aria-label={t(`layout.corner.${c}`)}
                  onClick={() => store().movePip(c)}
                />
              ))}
            </div>
          </div>
        ) : null}
      </main>
      <nav className="tabs" aria-label={t("layout.tabs")} data-testid="mobile-tabs">
        {VIEW_IDS.map((v) => {
          const Icon = VIEW_REGISTRY[v].icon;
          return (
            <button
              key={v}
              type="button"
              className="tab"
              aria-current={v === main ? "page" : undefined}
              data-testid={`tab-${v}`}
              onClick={() => store().showOnly(v)}
            >
              <Icon aria-hidden="true" size={18} />
              <span>{t(VIEW_REGISTRY[v].titleKey)}</span>
            </button>
          );
        })}
      </nav>
      <div className="tabs-gap" aria-hidden="true" />
    </>
  );
}
