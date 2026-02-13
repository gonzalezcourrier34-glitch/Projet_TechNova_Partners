from pathlib import Path

modules1 = {
    "app.models": "app_models.md",
    "app.api": "app_api.md",
    "domain.domain": "domain.md",
    "service.technova_service": "technova_service.md"
}
modules2 = {
    "app.database": "database.md"
}

modules3 = {
    "app.security": "security.md"
}

docs_dir1 = Path("docs/api_reference")
docs_dir1.mkdir(parents=True, exist_ok=True)

for module, filename in modules1.items():
    content = f"# {module}\n\n::: {module}\n"
    (docs_dir1 / filename).write_text(content, encoding="utf-8")

docs_dir2 = Path("docs/init_base")
docs_dir2.mkdir(parents=True, exist_ok=True)

for module, filename in modules2.items():
    content = f"# {module}\n\n::: {module}\n"
    (docs_dir2 / filename).write_text(content, encoding="utf-8")

docs_dir3 = Path("docs/securite")
docs_dir3.mkdir(parents=True, exist_ok=True)

for module, filename in modules3.items():
    content = f"# {module}\n\n::: {module}\n"
    (docs_dir3 / filename).write_text(content, encoding="utf-8")

print(" Fichiers MkDocs API générés.")
