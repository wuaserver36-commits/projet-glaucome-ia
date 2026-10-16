"""
Grille du champ visuel Humphrey 24-2 et conversion œil gauche -> œil droit.

Convention de saisie des 52 valeurs (colonnes TD_01 à TD_52) :
on lit la carte de déviation totale TELLE QU'ELLE EST IMPRIMÉE,
ligne par ligne de haut en bas, et de gauche à droite dans chaque ligne,
en sautant les 2 cases de la tache aveugle.
  - Œil droit : tache aveugle à droite de l'impression.
  - Œil gauche : tache aveugle à gauche de l'impression.
Le programme se charge ensuite de remettre les yeux gauches dans
l'orientation de l'œil droit, pour que chaque colonne corresponde
toujours à la même zone anatomique.
"""
import numpy as np

# Coordonnées (en degrés) des points du 24-2 pour un œil droit, ligne par ligne.
# x négatif = champ nasal, x positif = champ temporal (côté tache aveugle).
_LIGNES_OD = {
    21: [-9, -3, 3, 9],
    15: [-15, -9, -3, 3, 9, 15],
    9: [-21, -15, -9, -3, 3, 9, 15, 21],
    3: [-27, -21, -15, -9, -3, 3, 9, 15, 21],
    -3: [-27, -21, -15, -9, -3, 3, 9, 15, 21],
    -9: [-21, -15, -9, -3, 3, 9, 15, 21],
    -15: [-15, -9, -3, 3, 9, 15],
    -21: [-9, -3, 3, 9],
}
TACHE_AVEUGLE_OD = {(15, 3), (15, -3)}


def _points(lignes, tache):
    return [(x, y)
            for y in sorted(lignes, reverse=True)
            for x in sorted(lignes[y])
            if (x, y) not in tache]


POINTS_OD = _points(_LIGNES_OD, TACHE_AVEUGLE_OD)
_LIGNES_OG = {y: [-x for x in xs] for y, xs in _LIGNES_OD.items()}
POINTS_OG = _points(_LIGNES_OG, {(-x, y) for x, y in TACHE_AVEUGLE_OD})
assert len(POINTS_OD) == len(POINTS_OG) == 52

COLONNES_TD = [f"TD_{i:02d}" for i in range(1, 53)]

# Pour chaque point k d'un œil gauche, indice du point équivalent chez l'œil droit.
_INDEX_OD = {p: i for i, p in enumerate(POINTS_OD)}
_OG_VERS_OD = np.array([_INDEX_OD[(-x, y)] for x, y in POINTS_OG])


def og_vers_od(valeurs_og):
    """Réordonne des valeurs lues sur un CV d'œil gauche dans l'ordre de l'œil droit."""
    v = np.asarray(valeurs_og, dtype=float)
    sortie = np.empty_like(v)
    sortie[..., _OG_VERS_OD] = v
    return sortie


def od_vers_og(valeurs_od):
    """Opération inverse (utilisée par le générateur de données fictives)."""
    v = np.asarray(valeurs_od, dtype=float)
    return v[..., _OG_VERS_OD]
