import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import app
from app.database import get_db
from app.models import Base
from app.security import verify_api_key

os.environ.setdefault("API_KEY", "ci-test-key")

TEST_DB_URL = "sqlite+pysqlite:///:memory:"

# On définit une fixture pour créer une base de données en mémoire pour les tests, 
# ce qui permet d'avoir un environnement de test isolé et rapide.
@pytest.fixture(scope="session")
def engine():
    eng = (
        create_engine(
            TEST_DB_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        .execution_options(schema_translate_map={"raw": None, "clean": None, "app": None})
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()

# On définit une fonction pour générer un payload d'exemple, 
# qui correspond aux features attendues par le modèle de prédiction.
@pytest.fixture()
def db(engine):
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# On définit une fixture pour créer un client de test FastAPI, 
# avec la sécurité désactivée pour les tests de routes,
def _override_db(db):
    def override_get_db():
        yield db
    return override_get_db

# On définit une fixture pour créer un client de test FastAPI,
# avec la sécurité activée pour les tests de sécurité, ce qui permet de tester les comportements d'authentification.
@pytest.fixture()
def client(db):
    # Sécurité OFF pour les tests routes
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[get_db] = _override_db(db)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()

# On définit une fixture pour créer un client de test FastAPI avec la sécurité activée,
# ce qui permet de tester les comportements d'authentification, notamment que les routes protégées nécessitent une clé API valide.
@pytest.fixture()
def secure_client(db):
    # Sécurité ON pour les tests security
    app.dependency_overrides.pop(verify_api_key, None)
    app.dependency_overrides[get_db] = _override_db(db)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
