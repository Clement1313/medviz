# -*- coding: utf-8 -*-
"""
RetinaScan AI — Interface d'analyse ophtalmoscopique
Portage fidèle du projet Figma Make (React / App.tsx) vers Plotly Dash.

Lancement :
    pip install dash
    python app.py
Puis ouvrir http://127.0.0.1:8050

Aucune dépendance autre que `dash`. Les icônes (lucide) sont intégrées en SVG inline.
La détection est simulée (mêmes données que la maquette d'origine) ; remplacez
`MOCK_EXUDATES` / la logique du callback `tick` par votre vrai modèle.
"""

import base64
import os
import time
from datetime import datetime
import requests

from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    ctx,
    ALL,
)
from dash.exceptions import PreventUpdate

# ----------------------------------------------------------------------------
# Données & logique métier (identiques à la maquette)
# ----------------------------------------------------------------------------

MOCK_EXUDATES = [
    {
        "id": 1,
        "x": 38,
        "y": 42,
        "radius": 3.2,
        "confidence": 94,
        "severity": "élevé",
        "type": "Mou",
        "size": "Grand",
    },
    {
        "id": 2,
        "x": 52,
        "y": 58,
        "radius": 2.1,
        "confidence": 87,
        "severity": "modéré",
        "type": "Dur",
        "size": "Grand",
    },
    {
        "id": 3,
        "x": 61,
        "y": 35,
        "radius": 1.8,
        "confidence": 91,
        "severity": "élevé",
        "type": "Dur",
        "size": "Petit",
    },
    {
        "id": 4,
        "x": 45,
        "y": 68,
        "radius": 1.4,
        "confidence": 78,
        "severity": "faible",
        "type": "Dur",
        "size": "Petit",
    },
    {
        "id": 5,
        "x": 70,
        "y": 52,
        "radius": 2.6,
        "confidence": 96,
        "severity": "élevé",
        "type": "Mou",
        "size": "Grand",
    },
    {
        "id": 6,
        "x": 29,
        "y": 61,
        "radius": 1.1,
        "confidence": 72,
        "severity": "faible",
        "type": "Dur",
        "size": "Petit",
    },
    {
        "id": 7,
        "x": 58,
        "y": 72,
        "radius": 1.9,
        "confidence": 83,
        "severity": "modéré",
        "type": "Mou",
        "size": "Petit",
    },
]

ANALYSIS_STEPS = [
    "Prétraitement de l'image...",
    "Segmentation rétinienne...",
    "Analyse des anomalies...",
    "Localisation des exsudats...",
    "Calcul des confidences...",
    "Génération du rapport...",
]

SEVERITY_COLORS = {"faible": "#facc15", "modéré": "#f97316", "élevé": "#ef4444"}
_SEV_WEIGHT = {"élevé": 3, "modéré": 2, "faible": 1}


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", API_URL).rstrip("/")


def call_analyze_api(img_data_uri, filename):
    if not img_data_uri or "," not in img_data_uri:
        raise ValueError(
            f"Format d'image invalide: {img_data_uri[:50] if img_data_uri else None}"
        )

    header, b64data = img_data_uri.split(",", 1)
    raw_bytes = base64.b64decode(b64data)

    files = {"file": (filename, raw_bytes, "image/png")}
    response = requests.post(f"{API_URL}/analyze", files=files)
    response.raise_for_status()
    data = response.json()
    return data["results"], data["diagnosis"]


def sort_results(data):
    return sorted(data, key=lambda r: (-_SEV_WEIGHT[r["severity"]], -r["confidence"]))


def build_report(results, image_name):
    sev = {
        k: sum(1 for r in results if r["severity"] == k)
        for k in ("faible", "modéré", "élevé")
    }
    avg = round(sum(r["confidence"] for r in results) / len(results))
    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    lines = [
        "========================================",
        "   RAPPORT D'ANALYSE RÉTINIENNE",
        "   RetinaScan AI — Modèle v2.4.1",
        "========================================",
        f"Date d'analyse   : {date}",
        f"Image analysée   : {image_name}",
        "",
        "--- RÉSULTATS ---",
        f"Exsudats rétiniens détectés : {len(results)}",
        f"  • Sévérité élevée  : {sev['élevé']}",
        f"  • Sévérité modérée : {sev['modéré']}",
        f"  • Sévérité faible  : {sev['faible']}",
        f"Confiance moyenne           : {avg}%",
        "",
        "--- DÉTAIL DES DÉTECTIONS ---",
        *[
            # f"  #{i + 1}  Sévérité: {r['severity']:<6} | Type: {r['type']:<3}, {r['size']:<5} "
            f"| Diamètre: {r['radius']:.1f} mm |"
            for i, r in enumerate(results)
        ],
        "",
        "--- AVIS CLINIQUE ---",
        f"{sev['élevé']} exsudat(s) de sévérité élevée détecté(s).",
        "Consultation ophtalmologique recommandée.",
        "========================================",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Icônes lucide (SVG inline -> data URI), zéro dépendance externe
# ----------------------------------------------------------------------------

ICONS = {
    "upload": '<path d="M12 3v12"/><path d="m17 8-5-5-5 5"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>',
    "scan-line": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "trash-2": '<path d="M10 11v6"/><path d="M14 11v6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    "zoom-in": '<circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/><line x1="11" x2="11" y1="8" y2="14"/><line x1="8" x2="14" y1="11" y2="11"/>',
    "zoom-out": '<circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/><line x1="8" x2="14" y1="11" y2="11"/>',
    "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "download": '<path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "eye": '<path d="M2 12c0 0 5.5-8 10-8s10 8 10 8-5.5 8-10 8-10-8-10-8z"/><circle cx="12" cy="12" r="3"/>',
}


def _data_uri(name, size, color, sw=2, cap="round", join="round"):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linecap="{cap}" stroke-linejoin="{join}">{ICONS[name]}</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def icon(name, size=16, color="#94a3b8", sw=2, cap="round", join="round", style=None):
    s = {
        "width": f"{size}px",
        "height": f"{size}px",
        "display": "block",
        "flexShrink": 0,
    }
    if style:
        s.update(style)
    return html.Img(src=_data_uri(name, size, color, sw, cap, join), style=s)


# Palette (slate / green) reprise du Tailwind d'origine
C = {
    "bg": "#0a0e14",
    "panel": "#0d1117",
    "b1": "#1e293b",
    "b2": "#334155",
    "b3": "#475569",
    "s200": "#e2e8f0",
    "s300": "#cbd5e1",
    "s400": "#94a3b8",
    "s500": "#64748b",
    "green": "#22c55e",
    "green600": "#16a34a",
    "red": "#f87171",
}

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

app = Dash(__name__, title="RetinaScan AI", suppress_callback_exceptions=True)
server = app.server


@app.callback(
    Output("state-history", "data"),
    Input("url", "pathname"),
    Input("btn-history", "n_clicks"),
)
def load_history(_pathname, _n):
    try:
        response = requests.get(f"{API_URL}/history")
        response.raise_for_status()
        data = response.json()
        print(f">>> Historique chargé: {len(data)} entrées")
        return data
    except requests.RequestException as e:
        print(f">>> Erreur chargement historique: {e}")
        return []


def dropzone_children():
    return html.Div(
        className="dz-inner",
        children=[
            html.Div(
                icon("upload", 24, C["s400"]), className="dz-icon", id="dz-icon-box"
            ),
            html.Div(
                [
                    html.P(
                        "Déposez votre capture rétinienne ici", className="dz-title"
                    ),
                    html.P("PNG, JPG, TIFF (≥ 2048×2048)", className="dz-sub"),
                ],
                className="dz-texts",
            ),
            html.Div(
                html.Span(
                    [icon("image", 16, C["s300"]), html.Span("Parcourir")],
                    className="btn-outline",
                ),
                className="dz-browse",
            ),
        ],
    )


app.layout = html.Div(
    className="root",
    children=[
        # ----- stores -----
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="state-image"),  # {"src":..., "name":...} | None
        dcc.Store(id="state-results"),  # [ ... ] | None
        dcc.Store(id="state-history", data=[]),
        dcc.Store(id="state-zoom", data=1),
        dcc.Store(
            id="state-analysis", data={"running": False, "step": 0, "progress": 0}
        ),
        dcc.Store(id="state-diagnosis"),
        dcc.Store(id="state-modal", data=False),
        dcc.Interval(id="analysis-interval", interval=450, disabled=True),
        dcc.Download(id="download-report"),
        # ----- logo (haut gauche) -----
        html.Div(
            icon("eye", 28, C["b3"], sw=1.5, cap="square", join="miter"),
            className="logo-eye",
        ),
        # ----- bouton historique (haut droite) -----
        html.Button(
            [icon("history", 16, C["s300"]), html.Span("Historique")],
            id="btn-history",
            className="history-btn",
            n_clicks=0,
        ),
        # ----- main -----
        html.Main(
            className="main",
            children=[
                # zone image
                html.Div(
                    id="image-box",
                    className="image-box empty",
                    children=[
                        dcc.Upload(
                            id="upload",
                            className="upload-zone",
                            children=dropzone_children(),
                            multiple=False,
                            accept="image/*",
                        ),
                        html.Div(
                            id="image-view",
                            className="image-view",
                            style={"display": "none"},
                        ),
                    ],
                ),
                # barre d'action
                html.Div(
                    className="action-bar",
                    children=[
                        html.Button(
                            [
                                icon("scan-line", 16, C["s500"]),
                                html.Span("Lancer l'analyse", id="analyze-label"),
                            ],
                            id="btn-analyze",
                            className="analyze-btn disabled",
                            disabled=True,
                            n_clicks=0,
                        ),
                        html.Div(id="analysis-status", className="analysis-status"),
                    ],
                ),
                # résultats
                html.Div(id="results-panel", style={"display": "none"}),
            ],
        ),
        # ----- modale historique -----
        html.Div(
            id="history-modal",
            className="modal-overlay",
            style={"display": "none"},
            children=html.Div(
                className="modal-card",
                children=[
                    html.Div(
                        className="modal-head",
                        children=[
                            html.Div(
                                [
                                    icon("history", 20, C["s400"]),
                                    html.H2(
                                        "Historique des analyses",
                                        className="modal-title",
                                    ),
                                ],
                                className="modal-head-left",
                            ),
                            html.Button(
                                icon("x", 20, C["s400"]),
                                id="btn-history-close",
                                className="modal-close",
                                n_clicks=0,
                            ),
                        ],
                    ),
                    html.Div(id="history-list", className="modal-body"),
                ],
            ),
        ),
    ],
)


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------


@app.callback(
    Output("state-image", "data"),
    Output("state-results", "data"),
    Output("state-zoom", "data"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    return {"src": contents, "name": filename or "capture"}, None, 1


@app.callback(
    Output("image-box", "className"),
    Output("upload", "style"),
    Output("image-view", "style"),
    Output("image-view", "children"),
    Input("state-image", "data"),
    Input("state-results", "data"),
    Input("state-zoom", "data"),
)
def render_box(img, results, zoom):
    if not img:
        return "image-box empty", {"display": "flex"}, {"display": "none"}, None
    return (
        "image-box filled",
        {"display": "none"},
        {"display": "flex"},
        build_image_view(img, results, zoom or 1),
    )


def build_image_view(img, results, zoom):
    markers = []
    for idx, ex in enumerate(results or []):
        d = max(16, ex["radius"] * 12)
        col = SEVERITY_COLORS[ex["severity"]]
        markers.append(
            html.Div(
                html.Span(str(idx + 1), className="marker-num"),
                className="marker",
                style={
                    "left": f"{ex['x']}%",
                    "top": f"{ex['y']}%",
                    "width": f"{d}px",
                    "height": f"{d}px",
                    "borderColor": col,
                    "backgroundColor": col + "15",
                },
            )
        )

    tools = [
        ("zoom-in", "btn-zoom-in", "Zoom +", C["s400"]),
        ("zoom-out", "btn-zoom-out", "Zoom -", C["s400"]),
        ("rotate-ccw", "btn-zoom-reset", "Réinitialiser", C["s400"]),
        ("trash-2", "btn-trash", "Supprimer", C["red"]),
    ]
    toolbar = html.Div(
        [
            html.Button(
                icon(ic, 14, col),
                id=bid,
                title=ttl,
                n_clicks=0,
                className="tool-btn danger" if bid == "btn-trash" else "tool-btn",
            )
            for ic, bid, ttl, col in tools
        ],
        className="toolbar",
    )

    badge = html.Div(
        [icon("image", 14, C["s400"]), html.Span(img["name"], className="badge-name")],
        className="file-badge",
    )
    zoom_label = html.Div(f"{round(zoom * 100)}%", className="zoom-label")

    circle = html.Div(
        [html.Img(src=img["src"], className="retina-img"), *markers],
        className="retina-circle",
    )
    scaled = html.Div(
        circle, className="retina-scale", style={"transform": f"scale({zoom})"}
    )

    return [badge, toolbar, zoom_label, scaled]


@app.callback(
    Output("state-zoom", "data", allow_duplicate=True),
    Input("btn-zoom-in", "n_clicks"),
    Input("btn-zoom-out", "n_clicks"),
    Input("btn-zoom-reset", "n_clicks"),
    State("state-zoom", "data"),
    prevent_initial_call=True,
)
def zoom_ctrl(zi, zo, zr, z):
    z = z or 1
    t = ctx.triggered_id
    if t == "btn-zoom-in" and zi:
        return min(round(z + 0.2, 2), 3)
    if t == "btn-zoom-out" and zo:
        return max(round(z - 0.2, 2), 0.5)
    if t == "btn-zoom-reset" and zr:
        return 1
    raise PreventUpdate


@app.callback(
    Output("state-image", "data", allow_duplicate=True),
    Output("state-results", "data", allow_duplicate=True),
    Output("state-zoom", "data", allow_duplicate=True),
    Input("btn-trash", "n_clicks"),
    prevent_initial_call=True,
)
def clear_image(_n):
    if not _n:
        raise PreventUpdate
    return None, None, 1


@app.callback(
    Output("state-analysis", "data"),
    Output("analysis-interval", "disabled"),
    Output("state-results", "data", allow_duplicate=True),
    Input("btn-analyze", "n_clicks"),
    State("state-image", "data"),
    State("state-analysis", "data"),
    prevent_initial_call=True,
)
def launch(_n, img, a):
    if not img or (a and a.get("running")):
        raise PreventUpdate
    return (
        {
            "running": True,
            "step": 0,
            "progress": 0,
            "animation_done": False,
            "api_called": False,
        },
        False,
        None,
    )


@app.callback(
    Output("state-analysis", "data", allow_duplicate=True),
    Output("analysis-interval", "disabled", allow_duplicate=True),
    Input("analysis-interval", "n_intervals"),
    State("state-analysis", "data"),
    prevent_initial_call=True,
)
def advance_animation(_n, a):
    """Fait uniquement avancer le compteur d'étapes, AUCUN appel réseau ici."""
    if not a or not a.get("running") or a.get("animation_done"):
        raise PreventUpdate

    n = len(ANALYSIS_STEPS)
    step = a["step"] + 1

    if step >= n:
        return {**a, "step": n, "progress": 100, "animation_done": True}, True

    return {**a, "step": step, "progress": round(step / n * 100)}, False


@app.callback(
    Output("state-analysis", "data", allow_duplicate=True),
    Output("state-results", "data", allow_duplicate=True),
    Output("state-diagnosis", "data"),
    Output("state-history", "data"),
    Input("state-analysis", "data"),
    State("state-image", "data"),
    State("state-history", "data"),
    prevent_initial_call=True,
)
def run_api_call(a, img, hist):
    if not a or not a.get("animation_done") or a.get("api_called"):
        raise PreventUpdate

    try:
        api_results, diagnosis = call_analyze_api(img["src"], img["name"])
        results = sort_results(api_results)
    except Exception as e:
        print(f">>> ERREUR: {e}")
        results, diagnosis = [], None

    entry = {
        "id": str(time.time()),
        "src": img["src"],
        "name": img["name"],
        "date": int(time.time() * 1000),
        "results": results,
        "diagnosis": diagnosis,
    }
    return (
        {"running": False, "step": a["step"], "progress": 100, "api_called": True},
        results,
        diagnosis,
        [entry] + (hist or []),
    )


@app.callback(
    Output("btn-analyze", "className"),
    Output("btn-analyze", "disabled"),
    Output("analyze-label", "children"),
    Output("analysis-status", "children"),
    Input("state-analysis", "data"),
    Input("state-results", "data"),
    Input("state-image", "data"),
)
def render_action(a, results, img):
    running = bool(a and a.get("running"))
    disabled = (not img) or running
    cls = "analyze-btn disabled" if disabled else "analyze-btn"
    label = "Analyse en cours..." if running else "Lancer l'analyse"

    if running:
        step = min(a["step"], len(ANALYSIS_STEPS) - 1)
        status = html.Div(
            [
                html.Div(
                    [
                        html.Span(ANALYSIS_STEPS[step]),
                        html.Span(f"{a['progress']}%", className="progress-pct"),
                    ],
                    className="progress-row",
                ),
                html.Div(
                    html.Div(
                        className="progress-fill", style={"width": f"{a['progress']}%"}
                    ),
                    className="progress-track",
                ),
            ],
            className="progress-wrap",
        )
    elif results:
        status = html.Div(
            [
                icon("check", 16, C["green"]),
                html.Span("Analyse terminée", className="done-txt"),
            ],
            className="status-done",
        )
    elif img:
        status = html.P(
            "Prêt pour l'analyse. Lancez la détection.", className="status-ready"
        )
    else:
        status = None
    return cls, disabled, label, status


STAGE_COLORS = {
    "Léger / NPDR précoce": C["green"],
    "Modéré / NPDR avancé": "#f97316",
    "Sévère / proche PDR": C["red"],
}


@app.callback(
    Output("results-panel", "children"),
    Output("results-panel", "style"),
    Input("state-diagnosis", "data"),
)
def render_diagnosis(diagnosis):
    if not diagnosis:
        return None, {"display": "none"}

    stage = diagnosis["stage"]
    color = STAGE_COLORS.get(stage, C["s400"])

    # score = max(0, min(100, int((1 - diagnosis["surface_ratio"]) * 100)))

    panel = html.Div(
        className="diagnosis-card",
        children=[
            # Header
            html.Div(
                className="diagnosis-header",
                children=[
                    html.Div([html.P("Analyse rétinienne terminée")]),
                    html.Div(
                        stage,
                        className="diagnosis-badge",
                        style={"background": color, "color": "white"},
                    ),
                ],
            ),
            # Interprétation
            html.Div(
                className="diagnosis-section",
                children=[
                    html.H4("Interprétation"),
                    html.P(diagnosis["interpretation"]),
                ],
            ),
            # Score IA
            html.Div(
                className="diagnosis-section",
                children=[
                    html.Div(
                        [
                            # html.Div(f"{score}%", className="score-value"),
                            # html.Div("Indice de confiance", className="score-label")
                        ],
                        className="score-circle",
                    )
                ],
            ),
            # Statistiques
            html.Div(
                className="stats-grid",
                children=[
                    html.Div(
                        className="stat-card",
                        children=[
                            html.H5("Exsudats"),
                            html.H2(str(diagnosis["n_exudates"])),
                        ],
                    ),
                    html.Div(
                        className="stat-card",
                        children=[
                            html.H5("Surface"),
                            html.H2(f"{diagnosis['surface_ratio'] * 100:.2f}%"),
                        ],
                    ),
                    # html.Div(
                    #     className="stat-card",
                    #     children=[
                    #         html.H5("Stade"),
                    #         html.H2(stage, style={"color": color})
                    #     ]
                    # ),
                ],
            ),
            # Recommandation
            # html.Div(
            #     className="recommendation-box",
            #     children=[
            #         html.H4("Recommandation"),
            #         html.P(
            #             "Une consultation ophtalmologique est recommandée afin de confirmer le diagnostic et définir la prise en charge adaptée."
            #         )
            #     ]
            # )
        ],
    )

    return panel, {"display": "block"}


@app.callback(
    Output("download-report", "data"),
    Input("btn-export", "n_clicks"),
    State("state-results", "data"),
    State("state-image", "data"),
    prevent_initial_call=True,
)
def export_report(_n, results, img):
    if not _n or not results:
        raise PreventUpdate
    text = build_report(results, img["name"] if img else "capture")
    return dict(content=text, filename=f"rapport_retinascan_{int(time.time())}.txt")


@app.callback(
    Output("state-modal", "data"),
    Input("btn-history", "n_clicks"),
    Input("btn-history-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(_o, _c):
    return ctx.triggered_id == "btn-history"


@app.callback(Output("history-modal", "style"), Input("state-modal", "data"))
def modal_style(is_open):
    return {"display": "flex" if is_open else "none"}


@app.callback(Output("history-list", "children"), Input("state-history", "data"))
def render_history(hist):
    if not hist:
        return html.Div(
            [
                icon(
                    "clock",
                    40,
                    C["s500"],
                    style={"opacity": 0.5, "marginBottom": "16px"},
                ),
                html.P("Aucune analyse dans l'historique.", className="empty-txt"),
            ],
            className="history-empty",
        )

    cards = []
    for item in hist:
        date_str = datetime.fromtimestamp(item["date"] / 1000).strftime(
            "%d/%m/%Y %H:%M:%S"
        )
        chips = []
        for sev, col in (
            ("élevé", "#ef4444"),
            ("modéré", "#f97316"),
            ("faible", "#facc15"),
        ):
            cnt = sum(1 for r in item["results"] if r["severity"] == sev)
            if cnt:
                chips.append(
                    html.Div(
                        [
                            html.Div(
                                className="chip-dot", style={"backgroundColor": col}
                            ),
                            html.Span(str(cnt), className="chip-num"),
                        ],
                        className="sev-chip",
                    )
                )
        cards.append(
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src=f"{API_URL}/history/{item['id']}/image",
                            className="hist-thumb-img",
                        )
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(item["name"], className="hist-name"),
                                    html.Span(date_str, className="hist-date"),
                                ],
                                className="hist-row-top",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "Total détectés", className="hist-k"
                                            ),
                                            html.Span(
                                                f"{len(item['results'])} exsudats",
                                                className="hist-v",
                                            ),
                                        ],
                                        className="hist-total",
                                    ),
                                    html.Div(className="hist-divider"),
                                    html.Div(chips, className="hist-chips"),
                                ],
                                className="hist-row-bottom",
                            ),
                        ],
                        className="hist-info",
                    ),
                ],
                className="hist-card",
                id={"type": "hist-item", "id": item["id"]},
                n_clicks=0,
            )
        )
    return html.Div(cards, className="history-grid")


@app.callback(
    Output("state-image", "data", allow_duplicate=True),
    Output("state-results", "data", allow_duplicate=True),
    Output("state-diagnosis", "data", allow_duplicate=True),
    Output("state-zoom", "data", allow_duplicate=True),
    Output("state-modal", "data", allow_duplicate=True),
    Input({"type": "hist-item", "id": ALL}, "n_clicks"),
    State("state-history", "data"),
    prevent_initial_call=True,
)
def restore_history(clicks, hist):
    if not ctx.triggered_id or not any(c for c in (clicks or [])):
        raise PreventUpdate
    tid = ctx.triggered_id["id"]
    item = next((h for h in (hist or []) if h["id"] == tid), None)
    if not item:
        raise PreventUpdate
    img_url = f"{API_URL}/history/{item['id']}/image"
    return (
        {"src": img_url, "name": item["name"]},
        item["results"],
        item.get("diagnosis"),
        1,
        False,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "8050")))
