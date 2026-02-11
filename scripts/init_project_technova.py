from scripts.create_db import main as create_db
from scripts.seed_from_csv import main as seed_csv
from scripts.build_ml_features import main as build_ml_features
from scripts.seed_ml_features import main as seed_ml_features

# Script d'initialisation du projet : crée la base, ingère les CSV, construit les features, etc.
def main():
    print("Initialisation du projet")

    print("Création / vérification du schéma")
    create_db()

    print("Ingestion des CSV")
    seed_csv()

    print("Construction table ML features")
    build_ml_features()

    print("Remplissage table ML features")
    seed_ml_features()

    print("Initialisation terminée")

if __name__ == "__main__":
    main()
