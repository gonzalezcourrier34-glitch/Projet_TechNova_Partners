"""
Gestion de la connexion à la base de données pour l’API TechNova Partners.

Ce module centralise :
- le chargement de la configuration de connexion à la base (DATABASE_URL),
- la création de l’engine SQLAlchemy,
- la configuration des sessions,
- la dépendance FastAPI permettant d’injecter une session DB dans les endpoints.

Objectif 
---------------------------------------
Séparer clairement la logique "base de données" du reste de l’application permet :
- d’éviter les duplications de code,
- de faciliter les tests (SQLite en mémoire),
- de rendre le projet plus lisible et plus professionnel.

Comportement selon l’environnement
----------------------------------
- En développement / production :
  - les variables sont chargées depuis un fichier `.env`
  - DATABASE_URL pointe généralement vers PostgreSQL

- En tests (pytest) :
  - pytest définit automatiquement `PYTEST_CURRENT_TEST`
  - le chargement du `.env` est désactivé
  - les tests peuvent fournir leur propre DATABASE_URL (ex: SQLite en mémoire)

Base supportée
--------------
- PostgreSQL (recommandé en production)
- SQLite (utilisé pour les tests et le développement rapide)

Le code est compatible SQLAlchemy 2.x.
"""
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
    """
    Fournit une session de base de données à un endpoint FastAPI.

    Cette fonction est utilisée comme dépendance (`Depends(get_db)`).

    Fonctionnement :
    - ouvre une session SQLAlchemy,
    - la met à disposition de l’endpoint,
    - garantit sa fermeture après la requête (même en cas d’erreur).

    Pourquoi ?
    ---------------------------
    - évite les fuites de connexions,
    - garantit une gestion propre du cycle de vie des sessions,
    - respecte les bonnes pratiques FastAPI + SQLAlchemy.

    Yields
    ------
    sqlalchemy.orm.Session
        Session active connectée à la base de données.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
