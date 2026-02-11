from fastapi import Header, HTTPException, status
import os


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
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
