from __future__ import annotations

import os
from typing import Generator

from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Charger les variables d'environnement depuis .env, sauf si on est dans un contexte de test 
# (pytest définit la variable d'environnement PYTEST_CURRENT_TEST)
if "PYTEST_CURRENT_TEST" not in os.environ:
    load_dotenv(find_dotenv())

# Si PYTEST_CURRENT_TEST est défini, on suppose que les tests gèrent eux-mêmes les variables 
# d'environnement nécessaires (par exemple via des fixtures pytest)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")

# SQLAlchemy 2.0 recommande d'utiliser future=True pour préparer la transition vers la nouvelle API, 
# et pool_pre_ping=True pour éviter les erreurs de connexion "MySQL server has gone away"
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)

# sessionmaker est configuré pour ne pas faire de commit automatique, 
# ne pas faire de flush automatique, et ne pas expirer les objets après commit (expire_on_commit=False) 
# pour éviter les problèmes d'accès aux données après le commit
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# Dependency pour obtenir une session de base de données dans les endpoints FastAPI,
# avec gestion automatique de la fermeture de la session après usage
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
