#!/usr/bin/env bash
set -e

export PGHOST=127.0.0.1
export PGPORT=5432
export PGUSER=technova
export PGPASSWORD=technova
export PGDATABASE=technova

echo "Vérification de la base..."

# Attendre que Postgres soit prêt
for i in $(seq 1 60); do
  if pg_isready -h $PGHOST -p $PGPORT -U $PGUSER > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Postgres prêt."

# Vérifier si la table clean.ml_features_employees existe
TABLE_EXISTS=$(psql -tAc "
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'clean'
    AND table_name = 'ml_features_employees'
);" || echo "false")

if [ "$TABLE_EXISTS" = "t" ]; then
  echo "Tables déjà présentes. Pas d'initialisation."
else
  echo "Tables absentes. Lancement init_project_technova.py..."
  poetry run python scripts/init_project_technova.py
  echo "Initialisation terminée."
fi
