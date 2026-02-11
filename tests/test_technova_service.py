import pytest
import numpy as np

from service.technova_service import TechNovaService

class FakeModel:
    def __init__(self, p1=0.8):
        self.p1 = float(p1)

    def predict_proba(self, X):
        return np.array([[1.0 - self.p1, self.p1]], dtype=float)

class FakeRequest:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data

def make_service(feature_columns, p1=0.8, threshold=0.5):
    # instancie sans __init__ (pas de joblib / fichiers)
    s = TechNovaService.__new__(TechNovaService)
    s.model = FakeModel(p1=p1)
    s.threshold = float(threshold)
    s.feature_columns = list(feature_columns)
    return s

def test_adapt_input_missing_feature_raises():
    s = make_service(feature_columns=["age", "revenu_mensuel"])

    req = FakeRequest({"age": 30})  # manque revenu_mensuel

    with pytest.raises(ValueError, match="Missing features"):
        s.adapt_input(req)

def test_predict_from_payload_applies_threshold():
    s = make_service(feature_columns=["age"], p1=0.8, threshold=0.5)

    req = FakeRequest({"age": 29})

    resp = s.predict_from_payload(req)

    assert resp.turnover_probability == 0.8
    assert resp.will_leave is True
    assert resp.employee_id is None

class FakeDBNoRow:
    def execute(self, *_a, **_kw):
        class R:
            def mappings(self):
                return self
            def first(self):
                return None
        return R()


def test_adapt_input_extra_feature_raises():
    s = make_service(feature_columns=["age", "revenu_mensuel"])

    req = FakeRequest({"age": 30, "revenu_mensuel": 2500, "foo": 123})

    with pytest.raises(ValueError, match="Unexpected features"):
        s.adapt_input(req)


def test_predict_from_payload_below_threshold():
    s = make_service(feature_columns=["age"], p1=0.2, threshold=0.5)

    req = FakeRequest({"age": 29})
    resp = s.predict_from_payload(req)

    assert resp.turnover_probability == 0.2
    assert resp.will_leave is False


def test_predict_from_clean_not_found_raises():
    # couvre la branche "aucune ligne clean"
    s = make_service(feature_columns=["age"], p1=0.8, threshold=0.5)
    s._sql_clean_latest = "SQL"  # juste pour éviter None

    with pytest.raises(ValueError):
        s.predict_from_clean(FakeDBNoRow(), employee_id=999)
