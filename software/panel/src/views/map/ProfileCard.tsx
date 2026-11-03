// The search profile card: what the missing person looked like, as the AI proposed it from
// the chat or as the pilot approved it. The pilot corrects the fields and approves; only
// then is the profile used. An observer sees the card but cannot change or approve it.
// The profile's text is never logged or toasted: it may hold health details.
import { Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { SearchProfile } from "../../api/types";
import { useMissionStore } from "../../state/missionStore";
import { useIsPilot } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import { Picker } from "../../ui/Picker";
import {
  baseOf,
  CATEGORIES,
  CERTAINTIES,
  COLORS,
  type EditableDraft,
  emptyDraft,
  FREE_TEXT_MAX,
  ITEM_KINDS,
  MAX_GARMENTS,
  MAX_ITEMS,
  PARTS,
  type ProfileBase,
  type ProfileState,
  pinPoint,
  shown,
  toApproval,
  toEditable,
  VISUAL_NOTES_MAX,
} from "./profile";

function Select<T extends string>({
  value,
  options,
  label,
  prefix,
  disabled,
  testId,
  onChange,
}: {
  value: T;
  options: readonly T[];
  label: string;
  prefix: string;
  disabled: boolean;
  testId: string;
  onChange: (v: T) => void;
}) {
  const { t } = useTranslation();
  return (
    <Picker
      value={value}
      options={options}
      label={label}
      labelOf={(o) => t(`${prefix}.${o}`)}
      disabled={disabled}
      testId={testId}
      onChange={onChange}
    />
  );
}

export function ProfileCard({ missionId }: { missionId: string }) {
  const { t } = useTranslation();
  const baseId = useId();
  const isPilot = useIsPilot();
  const rev = useMissionStore((s) => s.profileRev);
  const annotations = useMissionStore((s) => s.annotations);
  const pins = useMemo(
    () =>
      Object.values(annotations).flatMap((a) => {
        const p = a.kind === "pin" ? pinPoint(a.geometry) : null;
        return p ? [{ id: a.id, label: a.label, ...p }] : [];
      }),
    [annotations],
  );
  const [state, setState] = useState<ProfileState | null>(null);
  const [draft, setDraft] = useState<EditableDraft | null>(null);
  // What the draft was built from: the approval names it, so the ground refuses it when
  // something newer arrived. While the pilot edits, a new event never replaces the draft.
  const [base, setBase] = useState<ProfileBase>({ base_version: 0, proposal_id: null });
  const [dirty, setDirty] = useState(false);
  const [newer, setNewer] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apply = useCallback((s: ProfileState) => {
    setState(s);
    setBase(baseOf(s));
    const { profile } = shown(s);
    setDraft(profile ? toEditable(profile) : null);
    setDirty(false);
    setNewer(false);
  }, []);

  const load = useCallback(
    async (keepEdits: boolean) => {
      try {
        const s = await api<ProfileState>(paths.searchProfile(missionId));
        if (keepEdits) setNewer(true);
        else apply(s);
      } catch {
        setError(t("profile.loadFailed"));
      }
    },
    [missionId, t, apply],
  );

  // On open, and on every search_profile event (a new proposal or approval). The
  // pilot's unsaved corrections stay; the card says something newer is waiting.
  // biome-ignore lint/correctness/useExhaustiveDependencies: rev is the refetch signal
  useEffect(() => {
    void load(dirty);
  }, [load, rev]);

  const { profile, pending } = shown(state);
  const readOnly = !isPilot;

  async function approve(): Promise<void> {
    if (!draft || busy || !isPilot) return;
    setBusy(true);
    setError(null);
    try {
      await api<SearchProfile>(paths.searchProfile(missionId), {
        method: "POST",
        body: toApproval(draft, base),
      });
      useUiStore.getState().toast(t("profile.approvedToast"));
      await load(false);
    } catch (e) {
      if (e instanceof HttpError && e.code === "PROFILE_STALE") setNewer(true);
      setError(e instanceof HttpError ? e.message : t("profile.failed"));
    } finally {
      setBusy(false);
    }
  }

  function patch(p: Partial<EditableDraft>): void {
    if (!draft) return;
    setDraft({ ...draft, ...p });
    setDirty(true);
  }

  const status = !profile
    ? t("profile.none")
    : pending
      ? t("profile.pending")
      : t("profile.approved", { version: profile.profile_version });

  return (
    <section className="profile-card" data-testid="profile-card" aria-label={t("profile.title")}>
      <header>
        <strong>{t("profile.title")}</strong>
        <span className={pending ? "warn" : "muted"} data-testid="profile-status">
          {status}
        </span>
      </header>
      {newer ? (
        <div className="profile-newer" role="status" data-testid="profile-newer">
          <p className="warn">{t("profile.newer")}</p>
          <button
            type="button"
            className="btn btn-small btn-outline"
            data-testid="profile-show-newer"
            onClick={() => {
              setError(null);
              void load(false);
            }}
          >
            {t("profile.showNewer")}
          </button>
        </div>
      ) : null}
      {!draft ? (
        isPilot && pins.length > 0 ? (
          <button
            type="button"
            className="btn btn-outline"
            data-testid="profile-new"
            onClick={() => {
              const pin = pins[0];
              if (pin) {
                setDraft(emptyDraft(pin, new Date()));
                setDirty(true);
              }
            }}
          >
            <Plus aria-hidden="true" size={16} />
            {t("profile.new")}
          </button>
        ) : (
          <p className="muted">{isPilot ? t("profile.needPin") : t("profile.noneYet")}</p>
        )
      ) : (
        <form
          className="profile-form"
          onSubmit={(e) => {
            e.preventDefault();
            void approve();
          }}
        >
          <label htmlFor={`${baseId}-cat`}>{t("profile.category")}</label>
          <Picker
            id={`${baseId}-cat`}
            testId="profile-category"
            value={draft.subject_category}
            options={CATEGORIES}
            labelOf={(c) => t(`profile.categories.${c}`)}
            disabled={readOnly}
            onChange={(subject_category) => patch({ subject_category })}
          />

          <fieldset data-testid="profile-garments">
            <legend>{t("profile.garments")}</legend>
            {draft.garments.map((g, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id of their own
              <div key={i} className="profile-row" data-testid={`profile-garment-${i}`}>
                <Select
                  value={g.part}
                  options={PARTS}
                  prefix="profile.parts"
                  label={t("profile.part")}
                  testId={`profile-garment-${i}-part`}
                  disabled={readOnly}
                  onChange={(part) =>
                    patch({
                      garments: draft.garments.map((x, j) => (j === i ? { ...x, part } : x)),
                    })
                  }
                />
                <Select
                  value={g.colors[0]}
                  options={COLORS}
                  prefix="profile.colors"
                  label={t("profile.color")}
                  testId={`profile-garment-${i}-color`}
                  disabled={readOnly}
                  onChange={(c) =>
                    patch({
                      garments: draft.garments.map((x, j) =>
                        j === i
                          ? { ...x, colors: [c, ...x.colors.slice(1)] as typeof x.colors }
                          : x,
                      ),
                    })
                  }
                />
                <Select
                  value={g.certainty}
                  options={CERTAINTIES}
                  prefix="profile.certainty"
                  label={t("profile.certaintyLabel")}
                  testId={`profile-garment-${i}-certainty`}
                  disabled={readOnly}
                  onChange={(certainty) =>
                    patch({
                      garments: draft.garments.map((x, j) => (j === i ? { ...x, certainty } : x)),
                    })
                  }
                />
                {readOnly ? null : (
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label={t("profile.remove")}
                    onClick={() => patch({ garments: draft.garments.filter((_, j) => j !== i) })}
                  >
                    <Trash2 aria-hidden="true" size={16} />
                  </button>
                )}
              </div>
            ))}
            {readOnly || draft.garments.length >= MAX_GARMENTS ? null : (
              <button
                type="button"
                className="btn btn-small btn-outline"
                data-testid="profile-add-garment"
                onClick={() =>
                  patch({
                    garments: [
                      ...draft.garments,
                      { part: "upper", colors: ["unknown"], certainty: "unsure" },
                    ],
                  })
                }
              >
                <Plus aria-hidden="true" size={14} />
                {t("profile.addGarment")}
              </button>
            )}
          </fieldset>

          <fieldset data-testid="profile-items">
            <legend>{t("profile.items")}</legend>
            {draft.items.map((it, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id of their own
              <div key={i} className="profile-row" data-testid={`profile-item-${i}`}>
                <Select
                  value={it.kind}
                  options={ITEM_KINDS}
                  prefix="profile.itemKinds"
                  label={t("profile.itemKind")}
                  testId={`profile-item-${i}-kind`}
                  disabled={readOnly}
                  onChange={(kind) =>
                    patch({ items: draft.items.map((x, j) => (j === i ? { ...x, kind } : x)) })
                  }
                />
                <Select
                  value={it.colors?.[0] ?? "unknown"}
                  options={COLORS}
                  prefix="profile.colors"
                  label={t("profile.color")}
                  testId={`profile-item-${i}-color`}
                  disabled={readOnly}
                  onChange={(c) =>
                    patch({
                      items: draft.items.map((x, j) =>
                        j === i ? { ...x, colors: c === "unknown" ? [] : [c] } : x,
                      ),
                    })
                  }
                />
                <Select
                  value={it.certainty}
                  options={CERTAINTIES}
                  prefix="profile.certainty"
                  label={t("profile.certaintyLabel")}
                  testId={`profile-item-${i}-certainty`}
                  disabled={readOnly}
                  onChange={(certainty) =>
                    patch({ items: draft.items.map((x, j) => (j === i ? { ...x, certainty } : x)) })
                  }
                />
                {readOnly ? null : (
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label={t("profile.remove")}
                    onClick={() => patch({ items: draft.items.filter((_, j) => j !== i) })}
                  >
                    <Trash2 aria-hidden="true" size={16} />
                  </button>
                )}
              </div>
            ))}
            {readOnly || draft.items.length >= MAX_ITEMS ? null : (
              <button
                type="button"
                className="btn btn-small btn-outline"
                data-testid="profile-add-item"
                onClick={() =>
                  patch({
                    items: [...draft.items, { kind: "backpack", colors: [], certainty: "unsure" }],
                  })
                }
              >
                <Plus aria-hidden="true" size={14} />
                {t("profile.addItem")}
              </button>
            )}
          </fieldset>

          {isPilot && pins.length > 0 ? (
            <>
              <label htmlFor={`${baseId}-seen`}>{t("profile.lastSeen")}</label>
              <Picker
                id={`${baseId}-seen`}
                testId="profile-last-seen"
                value={
                  pins.find((p) => p.lat === draft.last_seen.lat && p.lon === draft.last_seen.lon)
                    ?.id ?? ""
                }
                options={["", ...pins.map((p) => p.id)]}
                labelOf={(id) =>
                  id === ""
                    ? t("profile.lastSeenAsProposed")
                    : (pins.find((p) => p.id === id)?.label ?? id)
                }
                disabled={false}
                onChange={(id) => {
                  const p = pins.find((x) => x.id === id);
                  if (p) patch({ last_seen: { ...draft.last_seen, lat: p.lat, lon: p.lon } });
                }}
              />
            </>
          ) : null}

          <label htmlFor={`${baseId}-notes`}>{t("profile.visualNotes")}</label>
          <input
            id={`${baseId}-notes`}
            className="input"
            data-testid="profile-visual-notes"
            maxLength={VISUAL_NOTES_MAX}
            value={draft.visual_notes}
            disabled={readOnly}
            onChange={(e) => patch({ visual_notes: e.target.value })}
          />

          <label htmlFor={`${baseId}-free`}>{t("profile.freeText")}</label>
          <textarea
            id={`${baseId}-free`}
            className="input"
            data-testid="profile-free-text"
            rows={3}
            maxLength={FREE_TEXT_MAX}
            value={draft.free_text}
            disabled={readOnly}
            aria-describedby={`${baseId}-free-warn`}
            onChange={(e) => patch({ free_text: e.target.value })}
          />
          <p id={`${baseId}-free-warn`} className="muted" data-testid="profile-free-warning">
            {t("profile.freeTextWarning")}
          </p>

          {error ? <p className="error">{error}</p> : null}
          {isPilot ? (
            <button
              type="submit"
              className="btn btn-primary"
              data-testid="profile-approve"
              disabled={busy}
            >
              {t("profile.approve")}
            </button>
          ) : (
            <p className="muted">{t("profile.pilotApproves")}</p>
          )}
        </form>
      )}
    </section>
  );
}
