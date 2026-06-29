# RetinaScan AI — Data (API)

> 🚧 **En stand-by** — service pas encore développé. Ce document sert de point de départ et sera complété au fur et à mesure de l'implémentation.

## Rôle prévu

Service backend (FastAPI) chargé de la gestion des données de l'application :

- stockage des images uploadées et de leurs métadonnées
- persistance de l'historique des analyses (actuellement géré côté front en mémoire, via `dcc.Store`)
- mise à disposition d'une API pour le [front](../front) et le service [traitement](../traitement)

## État actuel

- Squelette FastAPI minimal dans [app/main.py](app/main.py) (route `GET /` uniquement)
- ⚠️ Le squelette contient encore une erreur à corriger avant la première exécution (`lifespan` non défini)
- Aucune base de données / modèle de persistance branché pour le moment

## Prérequis

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)

## Lancement (une fois le squelette corrigé)

```bash
cd data
uv sync
uv run fastapi dev app/main.py
```

## Structure

```
data/
├── app/
│   └── main.py        # Point d'entrée FastAPI
├── pyproject.toml      # Dépendances (fastapi)
└── uv.lock
```

## À faire

- [ ] Corriger le squelette FastAPI (`main.py`)
- [ ] Choisir et configurer le stockage (base de données / fichiers)
- [ ] Définir le contrat d'API (endpoints, schémas de requête/réponse)
- [ ] Brancher le front sur ce service (historique persistant)
