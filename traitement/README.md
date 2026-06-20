# RetinaScan AI — Traitement (API)

## Structure

```
traitement/
├── app/
│   └── main.py        # Point d'entrée FastAPI
├── pyproject.toml      # Dépendances (fastapi)
└── uv.lock
```

## Lancer le back
```
uv run fastapi dev app/main.py --port 8000
```