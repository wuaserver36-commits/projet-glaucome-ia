"""
Détection précoce du glaucome par IA à partir du champ visuel Humphrey 24-2.
Programme conforme au protocole : modèle principal XGBoost sur les 52 valeurs
de déviation totale, comparé à une régression logistique (MD + PSD), à un
random forest, au GHT et aux lecteurs humains.

Utilisation :
    python modele_glaucome.py mes_donnees.csv --essais 100 --sortie resultats

Colonnes obligatoires du fichier CSV (une ligne par œil) :
    patient_id, oeil (OD ou OG), glaucome (0 = normal, 1 = glaucome débutant),
    TD_01 ... TD_52 (déviation totale en dB, voir grille_24_2.py pour l'ordre)
Colonnes facultatives :
    MD, PSD                     -> régression logistique de référence
    GHT                         -> "normal", "limite" ou "hors limites"
    lecteur_1, lecteur_2, ...   -> avis des lecteurs humains (0 ou 1)
    age, sexe, atcd_familial, pachymetrie -> modèle secondaire exploratoire
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest, norm
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grille_24_2 import COLONNES_TD, POINTS_OD, TACHE_AVEUGLE_OD, og_vers_od

GRAINE = 2026          # graine aléatoire fixe : résultats reproductibles
PART_TEST = 0.20       # 20 % des patients mis de côté pour l'examen final
CLINIQUE = ["age", "sexe_f", "atcd_familial", "pachymetrie"]


# ---------------------------------------------------------------- 1. Données
def nombre(serie):
    """Convertit une colonne en nombres, en acceptant la virgule décimale."""
    return pd.to_numeric(serie.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


def code_ght(valeur):
    """0 = normal, 1 = limite, 2 = hors limites normales, NaN si absent."""
    if pd.isna(valeur):
        return np.nan
    v = str(valeur).lower()
    if "hors" in v or "outside" in v:
        return 2
    if "dans" in v or "within" in v or "normal" in v:
        return 0
    if "limite" in v or "border" in v:
        return 1
    return np.nan


def charger(chemin):
    df = pd.read_csv(chemin, sep=None, engine="python", dtype=str)
    manquantes = [c for c in ["patient_id", "oeil", "glaucome", *COLONNES_TD] if c not in df]
    if manquantes:
        raise SystemExit(f"Colonnes manquantes dans le fichier : {manquantes}")

    for c in ["glaucome", *COLONNES_TD, "MD", "PSD", "age", "atcd_familial", "pachymetrie"]:
        if c in df:
            df[c] = nombre(df[c])
    for c in [c for c in df if c.startswith("lecteur_")]:
        df[c] = nombre(df[c])
    df["oeil"] = df["oeil"].str.upper().str.strip()

    avant = len(df)
    df = df[df["oeil"].isin(["OD", "OG"]) & df["glaucome"].isin([0, 1])]
    df = df.dropna(subset=COLONNES_TD)
    if len(df) < avant:
        print(f"  {avant - len(df)} ligne(s) écartée(s) : œil, diagnostic ou valeurs TD manquants.")

    og = df["oeil"] == "OG"
    df.loc[og, COLONNES_TD] = og_vers_od(df.loc[og, COLONNES_TD].to_numpy())

    # Un seul œil par patient, tiré au sort
    df = (df.groupby("patient_id", group_keys=False)
            .sample(n=1, random_state=GRAINE).reset_index(drop=True))
    df["glaucome"] = df["glaucome"].astype(int)
    if "sexe" in df:
        df["sexe_f"] = (df["sexe"].astype(str).str.upper().str[0] == "F").astype(int)
    if "GHT" in df:
        df["GHT_code"] = df["GHT"].map(code_ght)
    return df


# ---------------------------------------------------------------- 2. Modèles
def creer_xgb(params, poids_positifs):
    return xgb.XGBClassifier(**params, scale_pos_weight=poids_positifs,
                             eval_metric="logloss", tree_method="hist",
                             random_state=GRAINE, n_jobs=-1)


def regler_hyperparametres(X, y, n_essais):
    poids = (y == 0).sum() / (y == 1).sum()
    cv = StratifiedKFold(5, shuffle=True, random_state=GRAINE)

    def objectif(essai):
        params = {
            "max_depth": essai.suggest_int("max_depth", 2, 6),
            "learning_rate": essai.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": essai.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": essai.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": essai.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": essai.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": essai.suggest_float("reg_lambda", 0.0, 10.0),
        }
        p = cross_val_predict(creer_xgb(params, poids), X, y, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(y, p)

    def suivi(etude, essai):
        if (essai.number + 1) % 10 == 0:
            print(f"  essai {essai.number + 1}/{n_essais} - meilleure AUC : {etude.best_value:.3f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    etude = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=GRAINE))
    etude.optimize(objectif, n_trials=n_essais, callbacks=[suivi])
    return etude.best_params, etude.best_value, poids


def seuils_decision(y, p):
    """Seuils fixés sur la validation croisée, puis appliqués tels quels au test."""
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.minimum(thr, 1.0)
    spec = 1 - fpr
    seuils = {}
    for cible in (0.95, 0.90):
        i = np.argmax(np.where(spec >= cible, tpr, -1))
        seuils[f"specificite_{int(cible * 100)}"] = float(thr[i])
    seuils["youden"] = float(thr[np.argmax(tpr - fpr)])
    return seuils


# ---------------------------------------------------------------- 3. Statistiques
def ic_wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    marge = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - marge, centre + marge)


def sens_spec(y, positif):
    y, positif = np.asarray(y), np.asarray(positif)
    vp = int(((positif == 1) & (y == 1)).sum()); n_pos = int((y == 1).sum())
    vn = int(((positif == 0) & (y == 0)).sum()); n_neg = int((y == 0).sum())
    return {"sensibilite": vp / n_pos, "sensibilite_IC95": ic_wilson(vp, n_pos),
            "specificite": vn / n_neg, "specificite_IC95": ic_wilson(vn, n_neg)}


def auc_bootstrap(y, p, n=2000):
    rng = np.random.default_rng(GRAINE)
    y, p = np.asarray(y), np.asarray(p)
    valeurs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            valeurs.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), tuple(np.percentile(valeurs, [2.5, 97.5]))


def _rangs_moyens(x):
    ordre = np.argsort(x); z = x[ordre]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    r = np.empty(n); r[ordre] = t
    return r


def test_delong(y, p1, p2):
    """Compare deux AUC sur les mêmes observations (test de DeLong)."""
    y = np.asarray(y)
    p1, p2 = np.asarray(p1), np.asarray(p2)

    if len(y) != len(p1) or len(y) != len(p2):
        raise ValueError("y, p1 et p2 doivent avoir la même longueur.")

    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y doit contenir les deux classes 0 et 1.")

    if not np.isfinite(p1).all() or not np.isfinite(p2).all():
        raise ValueError("Les prédictions doivent être finies.")

    if np.array_equal(p1, p2):
        return 1.0

    s = np.vstack([p1, p2])
    pos, neg = s[:, y == 1], s[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]

    tx = np.array([_rangs_moyens(r) for r in pos])
    ty = np.array([_rangs_moyens(r) for r in neg])
    tz = np.array([
        _rangs_moyens(np.concatenate([pos[k], neg[k]]))
        for k in range(2)
    ])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n

    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    difference = aucs[0] - aucs[1]

    if variance <= 1e-15:
        return 1.0 if abs(difference) <= 1e-15 else float("nan")

    z = difference / np.sqrt(variance)
    return float(2 * norm.sf(abs(z)))


def test_mcnemar(y, a, b, groupe):
    """Compare deux méthodes sur les glaucomes (groupe=1) ou les normaux (groupe=0)."""
    y, a, b = map(np.asarray, (y, a, b))
    juste_a, juste_b = (a == y)[y == groupe], (b == y)[y == groupe]
    n01 = int((juste_a & ~juste_b).sum()); n10 = int((~juste_a & juste_b).sum())
    return float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def kappa_fleiss(avis):
    avis = np.asarray(avis); n, r = avis.shape
    positifs = avis.sum(axis=1)
    comptes = np.stack([r - positifs, positifs], axis=1)
    p_i = ((comptes**2).sum(axis=1) - r) / (r * (r - 1))
    p_j = comptes.sum(axis=0) / (n * r)
    return float((p_i.mean() - (p_j**2).sum()) / (1 - (p_j**2).sum()))


# ---------------------------------------------------------------- 4. Figures
def figure_roc(y, courbes, points, chemin):
    fig, ax = plt.subplots(figsize=(6, 6))
    for nom, p in courbes.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{nom} (AUC {roc_auc_score(y, p):.2f})")
    for nom, (fpr, tpr) in points.items():
        ax.plot(fpr, tpr, "o", label=nom)
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("1 - spécificité"); ax.set_ylabel("Sensibilité")
    ax.set_title("Courbes ROC sur le jeu de test"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)


def figure_calibration(y, p, chemin):
    obs, pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pred, obs, "o-", label="XGBoost (validation croisée)")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="Calibration parfaite")
    ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Proportion observée de glaucomes")
    ax.set_title("Calibration du modèle"); ax.legend()
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)


def figure_importance(importance, chemin):
    """Carte des points les plus discriminants, en orientation œil droit."""
    fig, ax = plt.subplots(figsize=(7, 6))
    pts = np.array(POINTS_OD)
    nuage = ax.scatter(pts[:, 0], pts[:, 1], c=importance, cmap="Reds", s=900, marker="s",
                       edgecolors="grey")
    for (x, y), v in zip(POINTS_OD, importance):
        ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7)
    for x, y in TACHE_AVEUGLE_OD:
        ax.scatter(x, y, s=900, marker="s", color="black")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-32, 32); ax.set_ylim(-26, 26); ax.set_aspect("equal")
    ax.set_xlabel("Nasal  <-  degrés  ->  Temporal")
    ax.set_title("Importance des points (|SHAP| moyen) - orientation œil droit")
    fig.colorbar(nuage, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 5. Programme
def main():
    ap = argparse.ArgumentParser(description="Détection du glaucome débutant par XGBoost (CV 24-2)")
    ap.add_argument("fichier", help="fichier CSV des données")
    ap.add_argument("--essais", type=int, default=100, help="essais de réglage (défaut : 100)")
    ap.add_argument("--sortie", default="resultats", help="dossier des résultats")
    a = ap.parse_args()
    sortie = Path(a.sortie); sortie.mkdir(exist_ok=True)

    print("1. Lecture des données")
    df = charger(a.fichier)
    n_glauc = int(df["glaucome"].sum())
    print(f"  {len(df)} patients retenus (un œil chacun) : {len(df) - n_glauc} normaux, {n_glauc} glaucomes")

    print("2. Mise de côté du jeu de test (20 %)")
    app, test = train_test_split(df, test_size=PART_TEST, stratify=df["glaucome"],
                                 random_state=GRAINE)
    X_app, y_app = app[COLONNES_TD].to_numpy(), app["glaucome"].to_numpy()
    X_test, y_test = test[COLONNES_TD].to_numpy(), test["glaucome"].to_numpy()
    print(f"  apprentissage : {len(app)} patients - test : {len(test)} patients")

    print(f"3. Réglage des hyperparamètres ({a.essais} essais, validation croisée en 5 blocs)")
    params, auc_cv, poids = regler_hyperparametres(X_app, y_app, a.essais)

    print("4. Seuils de décision et calibration (validation croisée)")
    cv = StratifiedKFold(5, shuffle=True, random_state=GRAINE)
    p_oof = cross_val_predict(creer_xgb(params, poids), X_app, y_app, cv=cv,
                              method="predict_proba")[:, 1]
    seuils = seuils_decision(y_app, p_oof)
    figure_calibration(y_app, p_oof, sortie / "calibration.png")

    print("5. Entraînement final et examen sur le jeu de test")
    modele = creer_xgb(params, poids).fit(X_app, y_app)
    modele.save_model(sortie / "modele_xgboost.json")
    p_test = modele.predict_proba(X_test)[:, 1]
    auc, ic = auc_bootstrap(y_test, p_test)

    res = {"effectifs": {"patients": len(df), "glaucomes": n_glauc,
                         "apprentissage": len(app), "test": len(test)},
           "hyperparametres": params, "AUC_validation_croisee": auc_cv,
           "seuils": seuils,
           "brier_validation_croisee": brier_score_loss(y_app, p_oof),
           "XGBoost": {"AUC": auc, "AUC_IC95": ic,
                       **{nom: sens_spec(y_test, (p_test >= s).astype(int))
                          for nom, s in seuils.items()}},
           "comparateurs": {}}
    courbes = {"XGBoost (52 points)": p_test}
    points = {}
    positif_95 = (p_test >= seuils["specificite_95"]).astype(int)

    # Comparateur 1 : régression logistique MD + PSD
    if {"MD", "PSD"} <= set(df):
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
        lr.fit(app[["MD", "PSD"]], y_app)
        p_lr = lr.predict_proba(test[["MD", "PSD"]])[:, 1]
        auc_lr, ic_lr = auc_bootstrap(y_test, p_lr)
        res["comparateurs"]["Logistique MD+PSD"] = {
            "AUC": auc_lr, "AUC_IC95": ic_lr, "p_DeLong_vs_XGBoost": test_delong(y_test, p_test, p_lr)}
        courbes["Logistique MD + PSD"] = p_lr

    # Comparateur 2 : random forest sur les 52 points
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=GRAINE, n_jobs=-1).fit(X_app, y_app)
    p_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf, ic_rf = auc_bootstrap(y_test, p_rf)
    res["comparateurs"]["Random forest"] = {
        "AUC": auc_rf, "AUC_IC95": ic_rf, "p_DeLong_vs_XGBoost": test_delong(y_test, p_test, p_rf)}
    courbes["Random forest"] = p_rf

    # Comparateur 3 : GHT (analyse principale : "limite" = négatif ; sensibilité : = positif)
    if "GHT_code" in test and test["GHT_code"].notna().all():
        for nom, seuil_ght in [("GHT (limite = négatif)", 2), ("GHT (limite = positif)", 1)]:
            pos = (test["GHT_code"] >= seuil_ght).astype(int).to_numpy()
            r = sens_spec(y_test, pos)
            r["p_McNemar_sens_vs_XGBoost_spec95"] = test_mcnemar(y_test, positif_95, pos, 1)
            res["comparateurs"][nom] = r
            points[nom] = (1 - r["specificite"], r["sensibilite"])

    # Comparateur 4 : lecteurs humains
    lecteurs = sorted(c for c in test if c.startswith("lecteur_"))
    if lecteurs and test[lecteurs].notna().all().all():
        for c in lecteurs:
            avis = test[c].astype(int).to_numpy()
            r = sens_spec(y_test, avis)
            r["p_McNemar_sens_vs_XGBoost_spec95"] = test_mcnemar(y_test, positif_95, avis, 1)
            r["p_McNemar_spec_vs_XGBoost_spec95"] = test_mcnemar(y_test, positif_95, avis, 0)
            res["comparateurs"][c] = r
            points[c.replace("_", " ").capitalize()] = (1 - r["specificite"], r["sensibilite"])
        if len(lecteurs) >= 2:
            res["kappa_fleiss_lecteurs"] = kappa_fleiss(test[lecteurs].astype(int))

    # Modèle secondaire exploratoire : 52 points + données cliniques simples
    cliniques = [c for c in CLINIQUE if c in df]
    if cliniques:
        cols = COLONNES_TD + cliniques
        m2 = creer_xgb(params, poids).fit(app[cols].to_numpy(dtype=float), y_app)
        p_m2 = m2.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]
        auc_m2, ic_m2 = auc_bootstrap(y_test, p_m2)
        res["modele_secondaire"] = {"variables_ajoutees": cliniques, "AUC": auc_m2, "AUC_IC95": ic_m2,
                                    "p_DeLong_vs_XGBoost": test_delong(y_test, p_test, p_m2)}
        courbes["XGBoost + clinique (exploratoire)"] = p_m2

    print("6. Importance des points (valeurs SHAP)")
    contributions = modele.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)[:, :-1]
    importance = np.abs(contributions).mean(axis=0)
    figure_importance(importance, sortie / "importance_points.png")
    figure_roc(y_test, courbes, points, sortie / "courbes_roc.png")
    pd.DataFrame({"colonne": COLONNES_TD, "x_deg": [p[0] for p in POINTS_OD],
                  "y_deg": [p[1] for p in POINTS_OD], "importance_SHAP": importance}
                 ).sort_values("importance_SHAP", ascending=False).to_csv(
                     sortie / "importance_points.csv", index=False)

    with open(sortie / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    ecrire_rapport(res, sortie / "rapport.txt")
    print(f"\nTerminé. Résultats dans le dossier « {sortie} ».\n")
    print((sortie / "rapport.txt").read_text(encoding="utf-8"))


def ecrire_rapport(res, chemin):
    pct = lambda v: f"{100 * v:.1f} %"
    ic = lambda t: f"[{pct(t[0])} ; {pct(t[1])}]"
    e = res["effectifs"]; x = res["XGBoost"]
    L = [
    "AVERTISSEMENT : données simulées. Résultats purement techniques, sans valeur clinique.",
    "",
    "RAPPORT - Détection du glaucome débutant par XGBoost (CV 24-2)",
    "=" * 62,
         f"Patients : {e['patients']} (dont {e['glaucomes']} glaucomes) - "
         f"apprentissage {e['apprentissage']}, test {e['test']}",
         f"AUC en validation croisée : {res['AUC_validation_croisee']:.3f} - "
         f"score de Brier : {res['brier_validation_croisee']:.3f}", "",
         "MODÈLE PRINCIPAL (jeu de test)",
         f"  AUC : {x['AUC']:.3f}  IC95 [{x['AUC_IC95'][0]:.3f} ; {x['AUC_IC95'][1]:.3f}]"]
    for nom in ("specificite_95", "specificite_90", "youden"):
        r = x[nom]
        L.append(f"  Seuil {nom} ({res['seuils'][nom]:.2f}) : sensibilité {pct(r['sensibilite'])} "
                 f"{ic(r['sensibilite_IC95'])}, spécificité {pct(r['specificite'])} "
                 f"{ic(r['specificite_IC95'])}")
    L += ["", "COMPARATEURS (jeu de test)"]
    for nom, r in res["comparateurs"].items():
        if "AUC" in r:
            L.append(f"  {nom} : AUC {r['AUC']:.3f} [{r['AUC_IC95'][0]:.3f} ; {r['AUC_IC95'][1]:.3f}]"
                     f" - DeLong vs XGBoost p = {r['p_DeLong_vs_XGBoost']:.3f}")
        else:
            L.append(f"  {nom} : sensibilité {pct(r['sensibilite'])}, spécificité "
                     f"{pct(r['specificite'])} - McNemar (sensibilité vs XGBoost à 95 % de "
                     f"spécificité) p = {r['p_McNemar_sens_vs_XGBoost_spec95']:.3f}")
    if "kappa_fleiss_lecteurs" in res:
        L.append(f"  Accord entre lecteurs (kappa de Fleiss) : {res['kappa_fleiss_lecteurs']:.2f}")
    if "modele_secondaire" in res:
        m = res["modele_secondaire"]
        L += ["", "MODÈLE SECONDAIRE EXPLORATOIRE (52 points + " + ", ".join(m["variables_ajoutees"]) + ")",
              f"  AUC {m['AUC']:.3f} [{m['AUC_IC95'][0]:.3f} ; {m['AUC_IC95'][1]:.3f}]"
              f" - DeLong vs modèle principal p = {m['p_DeLong_vs_XGBoost']:.3f}"]
    L += ["", "Figures : courbes_roc.png, calibration.png, importance_points.png"]
    Path(chemin).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
