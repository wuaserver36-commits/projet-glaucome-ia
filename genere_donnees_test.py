"""
Génère un fichier de DONNÉES FICTIVES au format attendu par modele_glaucome.py.

Sert uniquement à tester le programme et à voir le format du fichier à remplir.
Ces données sont simulées : les résultats obtenus avec elles n'ont aucune valeur
scientifique.

Utilisation :
    python genere_donnees_test.py --patients 700 --sortie donnees_test.csv
"""
import argparse

import numpy as np
import pandas as pd

from grille_24_2 import COLONNES_TD, POINTS_OD, od_vers_og


def simuler(n_patients, graine=0):
    rng = np.random.default_rng(graine)
    pts = np.array(POINTS_OD)
    x, y = pts[:, 0], pts[:, 1]
    ecc = np.hypot(x, y)
    zone_sup = (y > 0) & (ecc >= 8) & (ecc <= 26) & (x < 12)   # arciforme supérieure
    zone_inf = (y < 0) & (ecc >= 8) & (ecc <= 26) & (x < 12)   # arciforme inférieure
    ressaut = x <= -21                                         # ressaut nasal

    lignes = []
    for pid in range(1, n_patients + 1):
        glauc_patient = rng.random() < 0.45
        age = int(np.clip(rng.normal(56 if glauc_patient else 50, 12), 18, 90))
        sexe = rng.choice(["H", "F"])
        atcd = int(rng.random() < (0.30 if glauc_patient else 0.10))
        pachy = int(rng.normal(525 if glauc_patient else 540, 30))
        # au moins un œil atteint chez un patient glaucomateux
        atteints = {"OD": False, "OG": False}
        if glauc_patient:
            premier = rng.choice(["OD", "OG"])
            atteints[premier] = True
            atteints["OG" if premier == "OD" else "OD"] = rng.random() < 0.5

        for oeil in ("OD", "OG"):
            td = rng.normal(-0.3, 1.9, 52)
            if atteints[oeil]:
                choix = rng.random()
                zone = zone_sup if choix < 0.4 else zone_inf if choix < 0.8 else zone_sup | zone_inf
                severite = rng.uniform(0.15, 0.8)
                touche = zone & (rng.random(52) < 0.35 + 0.3 * ressaut)
                defaut = touche * rng.uniform(2, 11, 52) * severite
                while (td - defaut).mean() < -5.5:   # rester dans le glaucome débutant
                    defaut *= 0.85
                td = td - defaut
            td = np.round(td)
            md = round(float(td.mean()), 2)
            psd = round(float(np.std(td - np.median(td)) + 0.6), 2)
            diff = abs(td[zone_sup].mean() - td[zone_inf].mean())
            pire = td[zone_sup | zone_inf].min()
            ght = ("hors limites" if diff > 1.8 or pire <= -9
                   else "limite" if diff > 1.1 else "normal")
            label = int(atteints[oeil])

            ligne = {"patient_id": f"P{pid:04d}", "oeil": oeil, "glaucome": label,
                     "age": age, "sexe": sexe, "atcd_familial": atcd,
                     "pachymetrie": pachy, "MD": md, "PSD": psd, "GHT": ght}
            # trois lecteurs fictifs de niveaux différents
            for i, (sens, spec) in enumerate([(0.62, 0.84), (0.72, 0.88), (0.82, 0.91)], 1):
                ligne[f"lecteur_{i}"] = int(rng.random() < sens) if label else int(rng.random() > spec)
            valeurs = td if oeil == "OD" else od_vers_og(td)
            ligne.update(dict(zip(COLONNES_TD, valeurs.astype(int))))
            lignes.append(ligne)
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=700)
    ap.add_argument("--sortie", default="donnees_test.csv")
    a = ap.parse_args()
    df = simuler(a.patients)
    df.to_csv(a.sortie, index=False)
    print(f"{len(df)} champs visuels fictifs ({a.patients} patients) écrits dans {a.sortie}")
