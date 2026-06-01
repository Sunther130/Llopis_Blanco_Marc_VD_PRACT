"""Mòdul per a la definició de constants i configuracions globals."""


from pathlib import Path

MAIN_TITLE = "Què determina l'èxit d'un joc a Steam?"
TAG_SCALE_LOGARITMICA = "Logarítmica"
TAG_SCALE_LINEAL = "Lineal"

SCATTER_FILLCOLOR="#39C466"
HRECT_SCATTER_OPACITY=0.20
SCATTER_OPACITY_CIRCLE=0.5

BG = "#f4f7fb"
CARD_BG = "#dce8f5"
TEXT = "#2d3748"
ACCENT = "#1a6896"

TITLE_FONT = {
    "family": "ET Book",
    "style": "italic",
    "size": 30,
    "color": TEXT,
}
CHART_FONT = {
    "family": "ET Bembo",
    "size": 18,
}
TEXT_FONT = {
    "family": "ET Bembo",
    "fontSize": "20px",
    "color": TEXT,
    "maxWidth": "1000px",
    "margin": "0 auto",
    "lineHeight": "1.75",
    "textAlign": "justify",
    "marginBottom": "24px",
}

SECTION = {"padding": "48px 24px"}
DIVIDER = {"borderColor": "#B8B8B8", "margin": "0 auto", "maxWidth": "860px"}


TEMPLATE = 'plotly_white'
DATASET_PATH = Path('data/steam_dataset.csv')
DATASET_PARQUET_PATH = Path('data/steam_clean.parquet')
