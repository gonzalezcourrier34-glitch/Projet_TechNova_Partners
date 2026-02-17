import pytest
import numpy as np

from service.technova_service import TechNovaService

class FakeModel:
    # Un modèle factice qui retourne une probabilité fixe pour la classe positive (p1)
    def __init__(self, p1=0.8):
        self.p1 = float(p1)

    # La méthode predict_proba doit retourner un tableau de shape (n_samples, n_classes) avec les probabilités pour chaque classe.
    def predict_proba(self, X):
        return np.array([[1.0 - self.p1, self.p1]], dtype=float)

class FakeRequest:
    # Un objet factice pour simuler une requête avec un payload de données.
    def __init__(self, data):
        self._data = data

    # La méthode json() simule la méthode d'une requête qui retourne les données du payload sous forme de dictionnaire.
    def model_dump(self):
        return self._data

def make_service(feature_columns, p1=0.8, threshold=0.5):
    # instancie sans __init__ (pas de joblib / fichiers)
    s = TechNovaService.__new__(TechNovaService)
    # on configure manuellement les attributs nécessaires pour les tests, notamment le modèle factice, le seuil de décision et les colonnes de features attendues.
    s.model = FakeModel(p1=p1)
    s.threshold = float(threshold)
    s.feature_columns = list(feature_columns)
    return s

# On teste que si une feature attendue est manquante dans le payload, la méthode adapt_input lève une erreur, 
# ce qui est important pour garantir que les données d'entrée sont complètes et conformes aux attentes du modèle.
def test_adapt_input_missing_feature_raises():
    s = make_service(feature_columns=["age", "revenu_mensuel"])

    req = FakeRequest({"age": 30})  # manque revenu_mensuel

    with pytest.raises(ValueError, match="Missing features"):
        s.adapt_input(req)

# On teste que la méthode predict_from_payload applique correctement le seuil de décision pour déterminer la classe prédite (will_leave),
# ce qui est crucial pour que les prédictions soient interprétables et exploitables par les utilisateurs de l'API.
def test_predict_from_payload_applies_threshold():
    s = make_service(feature_columns=["age"], p1=0.8, threshold=0.5)

    req = FakeRequest({"age": 29})

    resp = s.predict_from_payload(req)

    assert resp.turnover_probability == 0.8
    assert resp.will_leave is True
    assert resp.employee_id is None

# On teste que si aucune ligne "clean" n'est trouvée pour un employee_id donné, la méthode predict_from_clean lève une erreur,
# ce qui est important pour éviter de faire des prédictions sur des données inexistantes ou incorrectes.
class FakeDBNoRow:
    def execute(self, *_a, **_kw):
        class R:
            def mappings(self):
                return self
            def first(self):
                return None
        return R()

# On teste que si des features inattendues sont présentes dans le payload, la méthode adapt_input lève une erreur,
# ce qui est important pour garantir que les données d'entrée sont conformes aux attentes du modèle et éviter des erreurs de traitement ou des prédictions incorrectes.
def test_adapt_input_extra_feature_raises():
    s = make_service(feature_columns=["age", "revenu_mensuel"])

    req = FakeRequest({"age": 30, "revenu_mensuel": 2500, "foo": 123})

    with pytest.raises(ValueError, match="Unexpected features"):
        s.adapt_input(req)

# On teste que si la probabilité de départ est inférieure au seuil, la classe prédite (will_leave) est False,
# ce qui correspond au comportement attendu du modèle de prédiction et permet aux utilisateurs de comprendre les résultats de l'API.
def test_predict_from_payload_below_threshold():
    s = make_service(feature_columns=["age"], p1=0.2, threshold=0.5)

    req = FakeRequest({"age": 29})
    resp = s.predict_from_payload(req)

    assert resp.turnover_probability == 0.2
    assert resp.will_leave is False

# On teste que si aucune ligne "clean" n'est trouvée pour un employee_id donné, la méthode predict_from_clean lève une erreur,
# ce qui est important pour éviter de faire des prédictions sur des données inexistantes ou incorrectes.
def test_predict_from_clean_not_found_raises():
    # couvre la branche "aucune ligne clean"
    s = make_service(feature_columns=["age"], p1=0.8, threshold=0.5)
    s._sql_clean_latest = "SQL"  # juste pour éviter None

    with pytest.raises(ValueError):
        s.predict_from_clean(FakeDBNoRow(), employee_id=999)
