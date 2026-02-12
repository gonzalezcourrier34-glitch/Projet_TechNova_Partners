import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Lancement de l'API FastAPI et du dashboard Streamlit en parallèle, avec gestion de l'arrêt propre
def run_api():
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.api:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ],
        cwd=ROOT,
    )

# Lancement du dashboard Streamlit, qui est un client HTTP de l'API (il envoie des requêtes, 
# il n'a pas besoin de lancer l'API lui-même)
def run_dashboard():
    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit",
            "run", str(ROOT / "dashboard" / "dshbd.py"),
            "--server.port", "8501",
        ],
        cwd=ROOT,
    )
# Le dashboard est un client HTTP, il envoie des requêtes à l'API, 
# mais il n'a pas besoin de lancer l'API lui-même.
def main():
    api = run_api()
    dashboard = run_dashboard()

    print("API: http://127.0.0.1:8000/docs")
    print("Dashboard: http://127.0.0.1:8501")

    try:
        while True:
            api_code = api.poll()
            dash_code = dashboard.poll()

            if api_code is not None:
                print(f"\nAPI stopped (code={api_code}). Stopping dashboard...")
                dashboard.terminate()
                break

            if dash_code is not None:
                print(f"\nDashboard stopped (code={dash_code}). Stopping API...")
                api.terminate()
                break

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nSTOP: arrêt demandé, fermeture propre...")
        api.terminate()
        dashboard.terminate()

if __name__ == "__main__":
    main()
