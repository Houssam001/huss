import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration de la page
st.set_page_config(page_title="Spotify Churn Analysis", layout="wide")

st.title("🎵 Spotify Churn Analysis")
st.markdown("""
Cette application présente une analyse du désabonnement (churn) des utilisateurs Spotify 
et les performances du modèle de prédiction associé.
""")

# Chargement des données
@st.cache_data
def load_data():
    # Assurez-vous que le fichier est dans le même répertoire
    return pd.read_csv("spotify_churn_dataset.csv")

df = load_data()

# --- Section 1 : Présentation du Dataset ---
st.header("📊 Exploration des Données")
if st.checkbox("Afficher les données brutes"):
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(10))

col1, col2 = st.columns(2)
with col1:
    st.write("**Statistiques Descriptives**")
    st.write(df.describe())
with col2:
    st.write("**Infos sur le Dataset**")
    st.write(f"Nombre de lignes : {df.shape[0]}")
    st.write(f"Nombre de colonnes : {df.shape[1]}")


# --- Section 2 : Visualisations Clés ---
st.header("📈 Visualisations")

# Élément interactif : Selectbox pour choisir la variable à visualiser
features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
selected_feature = st.selectbox("Choisissez une variable pour voir sa distribution :", features)

fig, ax = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax, color="green")
ax.set_title(f"Distribution de {selected_feature}")
st.pyplot(fig)



# --- Section 3 : Résultats du Modèle ---
st.header("🤖 Résultats du Modèle")

# Simulation de résultats (À remplacer par vos vraies métriques)
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Précision (Accuracy)", "85%", "+2%")
col_m2.metric("Rappel (Recall)", "78%", "-1%")
col_m3.metric("F1-Score", "81%")

st.info("Note : Le modèle utilise un algorithme de Logistic Regression pour prédire le Churn.")

