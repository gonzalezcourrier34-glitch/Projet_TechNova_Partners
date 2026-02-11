import os
import webbrowser

# les docs à générer sont celles des modules suivants, qui contiennent les fonctions et classes principales de l'app:
modules = [
    "app.models",
    "app.api",
    "domain.domain",
    "service.technova_service",
]

DOCS_DIR = "docs"

# Crée le dossier docs s'il n'existe pas
os.makedirs(DOCS_DIR, exist_ok=True)

# Se placer dans docs
os.chdir(DOCS_DIR)

# Génération avec pydoc
for m in modules:
    os.system(f"python -m pydoc -w {m}")

# Ouvre la doc principale
webbrowser.open("app.models.html")
webbrowser.open("app.api.html")
