# tests/test_api.py
import pytest
import numpy as np

from domain.domain import ModelResponse
from service.technova_service import TechNovaService


def sample_payload():
    return {
        "note_evaluation_precedente": 3,
        "niveau_hierarchique_poste": 2,
        "note_evaluation_actuelle": 3,
        "heures_supplementaires": 1,
        "augmentation_salaire_precedente": 11.0,
        "age": 41,
        "genre": 0,
        "revenu_mensuel": 5993,
        "statut_marital": "célibataire",
        "departement": "commercial",
        "poste": "cadre commercial",
        "nombre_experiences_precedentes": 8,
        "annee_experience_totale": 8,
        "annees_dans_l_entreprise": 6,
        "annees_dans_le_poste_actuel": 4,
        "nombre_participation_pee": 0,
        "nb_formations_suivies": 0,
        "distance_domicile_travail": 1,
        "niveau_education": 2,
        "domaine_etude": "infra & cloud",
        "frequence_deplacement": 1,
        "annees_depuis_la_derniere_promotion": 0,
        "annees_sous_responsable_actuel": 5,
        "satisfaction_moyenne": 2.0,
        "nonlineaire_participation_pee": 0.0,
        "ratio_heures_sup_salaire": 0.0001668335001668335,
        "nonlinaire_charge_contrainte": 0.008264462809917356,
        "nonlinaire_surmenage_insatisfaction": -1.0,
        "jeune_surcharge": 0,
        "anciennete_sans_promotion": 0.8571428571428571,
        "mobilite_carriere": 0.8888888888888888,
        "risque_global": -0.000143000143000143,
    }


@pytest.fixture(autouse=True)
def patch_service(monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    def fake_init(self):
        self.threshold = 0.5
        self.feature_columns = list(sample_payload().keys())
        self.model = FakeModel()

    def fake_predict_from_clean(self, db, employee_id: int):
        if employee_id == 1:
            return ModelResponse(employee_id=1, turnover_probability=0.7, will_leave=True)
        raise ValueError("not found")

    monkeypatch.setattr(TechNovaService, "__init__", fake_init)
    monkeypatch.setattr(TechNovaService, "predict_from_clean", fake_predict_from_clean)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200

def test_predict_by_features_ok(client):
    r = client.post("/predict/by-features", json=sample_payload())
    assert r.status_code == 200, r.text
    data = r.json()
    assert "turnover_probability" in data
    assert "will_leave" in data

def test_predict_by_id_200_and_404(client):
    assert client.post("/predict/by-id/1").status_code == 200
    assert client.post("/predict/by-id/2").status_code == 404

def test_latest_predictions_returns_list(client):
    client.post("/predict/by-features", json=sample_payload())
    r = client.get("/predictions/latest", params={"limit": 3})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)

## Obligaoitre pour couvrir le coverage de la branche
# tester que l'endpoint de prédiction par ID gère une exception inattendue et renvoie un HTTP 500.
def test_predict_by_id_returns_500_on_unexpected_error(client, monkeypatch):
    # Force la branche "except Exception" -> HTTP 500
    def boom(self, db, employee_id: int):
        raise Exception("boom")

    monkeypatch.setattr(TechNovaService, "predict_from_clean", boom)

    r = client.post("/predict/by-id/1")
    assert r.status_code == 500, r.text

# tester que l'endpoint de readiness vérifie la connexion à la DB et renvoie 503 si la table clean.ml_features_employees est manquante.
def test_ready_returns_503_when_clean_table_missing(client, monkeypatch):
    # Ici on override directement les dépendances FastAPI (get_db, get_service)
    import app.api as api

    class FakeService:
        model = object()  # "modèle chargé"

    class FakeDB:
        def execute(self, stmt, *_a, **_kw):
            s = str(stmt)
            if "FROM clean.ml_features_employees" in s:
                raise Exception("relation does not exist")
            return True

    def fake_get_service():
        return FakeService()

    def fake_get_db():
        yield FakeDB()

    api.app.dependency_overrides[api.get_service] = fake_get_service
    api.app.dependency_overrides[api.get_db] = fake_get_db

    try:
        r = client.get("/ready")
        assert r.status_code == 503, r.text
        assert "Clean table not ready" in r.text
    finally:
        api.app.dependency_overrides.clear()
