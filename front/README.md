# RetinaScan AI — Front (Dash)

Interface web d'analyse ophtalmoscopique. Portage du prototype Figma Make (React) vers [Plotly Dash](https://dash.plotly.com/).

## Fonctionnalités

- **Upload** d'une image de fond d'œil (PNG / JPG / TIFF)
- **Visualisation** de l'image avec contrôles de zoom (zoom +/-, réinitialisation, suppression)
- **Lancement d'une analyse** simulée (barre de progression par étapes : prétraitement, segmentation, détection des vaisseaux, etc.)
- **Résultats** : liste des exsudats détectés, triés par sévérité et confiance, avec marqueurs positionnés sur l'image
- **Export** du rapport d'analyse au format texte
- **Historique** des analyses précédentes (modale dédiée, restauration d'une analyse passée)

> La détection est actuellement **simulée** (`MOCK_EXUDATES` dans [app.py](app.py)). À remplacer par les appels au service `traitement` lorsque celui-ci sera prêt.

## Installation & lancement

Avec `uv` :

```bash
cd front
uv sync
uv run python app.py
```

Puis ouvrir [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Structure

```
front/
├── app.py            # Application Dash (layout, callbacks, logique métier)
├── assets/
│   └── style.css      # Styles (repris de la maquette Tailwind d'origine)
├── pyproject.toml     # Dépendances (dash)
└── uv.lock
```
