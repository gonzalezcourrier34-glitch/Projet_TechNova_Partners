import importlib
import sys


# On doit recharger le module database pour que les changements d'environnement soient pris en compte
def reload_database_module(monkeypatch, database_url: str | None):
    if database_url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database_url)

    sys.modules.pop("app.database", None)
    import app.database  # noqa: F401
    return importlib.reload(app.database)

# tester que si DATABASE_URL n'est pas défini, le module database utilise une URL SQLite en mémoire par défaut.
def test_database_defaults_to_sqlite_memory_when_env_missing(monkeypatch):
    db = reload_database_module(monkeypatch, None)
    assert hasattr(db, "DATABASE_URL")
    assert db.DATABASE_URL.startswith("sqlite+pysqlite:///")

# tester que si DATABASE_URL est défini, le module database l'utilise.
# Ici on ne teste pas la connexion à la DB, juste que la variable est bien prise en compte.
def test_get_db_closes_session(monkeypatch):
    db = reload_database_module(monkeypatch, "sqlite+pysqlite:///:memory:")

    class FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_session = FakeSession()

    # On remplace SessionLocal par une fonction qui retourne notre FakeSession
    monkeypatch.setattr(db, "SessionLocal", lambda: fake_session)

    gen = db.get_db()
    session = next(gen)

    assert session is fake_session

    # Ferme le generator => déclenche finally => close()
    gen.close()
    assert fake_session.closed is True
