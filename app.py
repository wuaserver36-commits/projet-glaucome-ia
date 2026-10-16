import streamlit as st
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Détection Glaucome - Champ Visuel", layout="wide")

st.title("👁️ Application de Détection Précoce du Glaucome")
st.write("Saisissez ou ajustez les valeurs de déviation totale de la grille 24-2 pour simuler un œil.")

# Création de curseurs pour quelques points clés à titre d'exemple
st.subheader("Valeurs des points du champ visuel")

valeurs_td = []
# On simule les 52 points avec une boucle simple pour l'interface
for i in range(1, 53):
    valeur = st.slider(f"TD_{i:02d} (dB)", min_value=-35.0, max_value=0.0, value=0.0, step=1.0)
    valeurs_td.append(valeur)

if st.button("Lancer l'analyse du modèle"):
    st.success("Les valeurs ont été récupérées avec succès !")
    # C'est ici qu'on branchera ton modèle XGBoost par la suite