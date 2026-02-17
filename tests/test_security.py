"""
teste l’endpoint /health avec la sécurité activée.

TEST 1 — Accès REFUSÉ sans clé API
TEST 2 — Accès AUTORISÉ avec une clé API valide

"""
import os

# on teste que l'accès à l'endpoint de santé nécessite une clé API valide.
def test_health_requires_api_key(secure_client):
    r = secure_client.get("/health")
    assert r.status_code == 401, r.text

# on teste que l'accès à l'endpoint de santé avec une clé API valide est autorisé.
def test_health_with_valid_api_key_is_ok(secure_client):
    api_key = os.getenv("API_KEY", "ci-test-key")
    r = secure_client.get("/health", headers={"X-API-Key": api_key})
    assert r.status_code == 200, r.text
