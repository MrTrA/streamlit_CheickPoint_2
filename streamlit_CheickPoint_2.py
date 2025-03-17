# Importez vos données et effectuez la phase d'exploration de base des données

import pandas as pd

inclusion_dataset = pd.read_csv('Financial_inclusion_dataset.csv')

# info générale du datase Elec_dataset

inclusion_dataset.info()

#from ydata_profiling import ProfileReport

# Generate the profile report
#profile = ProfileReport(inclusion_dataset, title='Rapport de Profilage Pandas')

# Display the report
#profile.to_notebook_iframe()
# Or generate an HTML report
#profile.to_file("rapport_inclusion_financière.html")

#  Encoder les caractéristiques catégorielles
from sklearn.preprocessing import LabelEncoder

# Encoder les données catégorielles
donnee_categorielle = inclusion_dataset.select_dtypes(include=['object']).columns
for col in donnee_categorielle:
# Créer un LabelEncoder pour chaque colonne catégorielle
    le = LabelEncoder()
    inclusion_dataset[col] = le.fit_transform(inclusion_dataset[col])

inclusion_dataset.info()

# Importer les bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Définir les caractéristiques (X) et la cible (y)
X = inclusion_dataset.drop('bank_account', axis=1)
y = inclusion_dataset['bank_account']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le classificateur (RandomForestClassifier ici)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = classifier.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy}")

print(classification_report(y_test, y_pred))

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Streamlit app
st.title("Prédiction de l'inclusion financière")

# Champs de saisie pour les fonctionnalités
input_data = {}
for column in X.columns:  # Utilisez les noms de colonnes de votre X
    input_data[column] = st.number_input(f"Entrer {column}", value=0)

# Créer un bouton pour déclencher la prédiction
if st.button("Prediction"):
    # Convertir les données d'entrée en DataFrame
    input_df = pd.DataFrame([input_data])

    # Faire des prédictions
    prediction = classifier.predict(input_df)

    # Afficher la prédiction
    st.write(f"Prediction: {prediction[0]}")  #En supposant une seule prédiction
