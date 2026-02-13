"""
Gestion de la sécurité de l’API TechNova Partners via une clé API (API Key).

Ce module contient la logique de protection des endpoints FastAPI à l’aide
d’un mécanisme simple mais efficace basé sur un header HTTP personnalisé.

Principe de sécurité
--------------------
- Chaque requête vers l’API doit inclure le header :
  
  X-API-Key: <API_KEY>

- La valeur attendue est définie côté serveur via la variable
  d’environnement `API_KEY`.

- Si la clé est absente, incorrecte ou si le serveur n’est pas configuré,
  l’accès est refusé.

Objectif
---------------------------------------
Ce mécanisme illustre :
- une première couche de sécurité côté API,
- l’utilisation des dépendances FastAPI (`Depends`) pour centraliser la logique,
- la différence entre :
  - une erreur de configuration serveur (500),
  - une erreur d’authentification client (401).

"""
from fastapi import Header, HTTPException, status
import os

# Sécurité : vérification de la clé API dans les requêtes entrantes
def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Vérifie la présence et la validité de la clé API dans une requête HTTP.

    Cette fonction est utilisée comme dépendance FastAPI
    (`Depends(verify_api_key)`) sur les endpoints protégés.

    Fonctionnement :
    ---------------
    1) Lit la clé API attendue depuis la variable d’environnement `API_KEY`
    2) Compare cette valeur avec celle fournie dans le header `X-API-Key`
    3) Refuse l’accès si :
       - la clé serveur n’est pas configurée,
       - la clé est absente,
       - la clé est incorrecte

    Paramètres
    ----------
    x_api_key : str | None
        Valeur du header HTTP `X-API-Key` envoyé par le client.

    Raises
    ------
    fastapi.HTTPException
        - 500 si la variable d’environnement API_KEY n’est pas définie côté serveur
        - 401 si la clé est absente ou invalide

    Returns
    -------
    None
        Ne retourne rien si l’authentification réussit.
        La requête peut alors continuer normalement.
    """
    api_key = os.getenv("API_KEY")  # lu à chaque requête (robuste)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured on server",
        )

    if x_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
