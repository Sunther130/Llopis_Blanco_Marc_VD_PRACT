"""Mòdul principal on es defineix l'estructura de l'aplicació i les callbacks de Dash."""


import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from settings import (
    DATASET_PARQUET_PATH,
    BG,
    CARD_BG,
    TEXT,
    ACCENT,
    TEXT_FONT,
    SECTION,
    DIVIDER,
    MAIN_TITLE
)

from clean import preprocess_data

from charts import (
    build_os_bar,
    build_genre_count_treemap,
    build_genre_sales_treemap,
    build_price_scatter,
    build_ccu_scatter,
    build_profile_scatter,
)


######################
# Data Preprocessing #
######################

if not DATASET_PARQUET_PATH.exists():
    preprocess_data()


################
# Data loading #
################
steam_df = pd.read_parquet(DATASET_PARQUET_PATH)
genre_cols = [c for c in steam_df.columns if c.startswith("Genre_")]
genres = steam_df.columns[steam_df.columns.str.startswith("Genre_")]

# Pre-compute genre_principal once
genre_sums = steam_df[genre_cols].sum()
steam_df["Genre_principal"] = steam_df[genre_cols].apply(
    lambda row: next(
        (
            c.replace("Genre_", "").replace("_", " ")
            for c in genre_sums.sort_values(ascending=False).index
            if row[c] > 0.5
        ),
        "Altres",
    ),
    axis=1,
)


##############################
# Pre-compute static figures #
##############################
fig_os = build_os_bar(steam_df)
fig_genre_count = build_genre_count_treemap(steam_df, genre_cols)
fig_genre_sales = build_genre_sales_treemap(steam_df, genre_cols)
fig_price = build_price_scatter(steam_df, genre_cols)
fig_ccu = build_ccu_scatter(steam_df, genre_cols)

n_games = f"{steam_df.shape[0]:,}"
year_min = int(steam_df["Release_year"].min())
year_max = int(steam_df["Release_year"].max())
n_genres = len(genres)
price_min = steam_df["Price"].min()
price_max = steam_df["Price"].max()


def stat_card(value: str, label: str):
    return dbc.Col(
        html.Div(
            [
                html.H2(value, style={"color": ACCENT, "margin": 0, "fontWeight": "bold"}),
                html.P(label, style={"color": TEXT, "margin": 0}),
            ],
            style={
                "textAlign": "center",
                "background": CARD_BG,
                "borderRadius": "8px",
                "padding": "20px 12px",
            },
        ),
        md=3,
        xs=6,
        style={"marginBottom": "12px"},
    )


def conclusion_card(title: str, body: str):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, style={"color": ACCENT}),
                html.P(body, style={"color": TEXT, "margin": 0}),
            ]
        ),
        style={**TEXT_FONT, "background": CARD_BG, "border": "none", "marginBottom": "12px"},
    )


##############
# App layout #
##############
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title=MAIN_TITLE,
)

app.layout = dbc.Container(
    fluid=True,
    style={"background": BG, "minHeight": "100vh", "fontFamily": "sans-serif"},
    children=[
        ##########
        # HEADER #
        ##########
        html.Div(
            style={"background": "#1b2838", "padding": "64px 24px 40px", "textAlign": "center",
                   "marginLeft": "calc(-1 * var(--bs-gutter-x, 0.75rem))",
                   "marginRight": "calc(-1 * var(--bs-gutter-x, 0.75rem))"},
            children=[
                html.H1(
                    MAIN_TITLE,
                    style={"color": "#66c0f4", "fontSize": "3rem", "fontWeight": "bold"},
                ),
                html.P(
                    "Una exploració del catàleg de videojocs més gran del món",
                    style={"color": "#c7d5e0", "fontSize": "1.2rem"},
                ),
                html.Hr(style={"borderColor": "#2a475e", "margin": "0 auto 0", "maxWidth": "860px"}),
                html.Div(
                    style={"height": "24px"},
                ),
                html.P(
                    [
                        "Aquest projecte presenta una anàlisi exploratòria del mercat dels videojocs a partir del catàleg de la plataforma Steam. "
                        "L'objectiu és examinar quins factors contribueixen a diferenciar els títols "
                        "dins d'un ecosistema especialment competitiu.",
                        html.Br(),
                        html.Br(),
                        "En particular, s'analitzen qüestions com el pes dels gèneres, la relació "
                        "entre preu i satisfacció, i l'associació entre popularitat i recepció per "
                        "part dels usuaris.",
                    ],
                    style={**TEXT_FONT, "color": "#c7d5e0", "maxWidth": "680px"},
                ),
            ],
        ),
        #########
        # STATS #
        #########
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    "El dataset que tractarem a continuació inclou més de 100.000 jocs publicats a la plataforma de vendes de videojocs de la companyia Valve "
                    " anomenada Steam, que ofereix una àmplia varietat de gèneres i preus. "
                    "Aquesta diversitat permet explorar com es distribueixen diferents "
                    "característiques al llarg del catàleg i com aquestes poden influir en la percepció dels usuaris.",
                    style={**TEXT_FONT},
                ),
                dbc.Row(
                    [
                        stat_card(n_games, "jocs al catàleg"),
                        stat_card(f"{year_min} - {year_max}", "anys coberts"),
                        stat_card(str(n_genres), "nombre de gèneres"),
                        stat_card(f"${price_min:.0f} - ${price_max:.0f}", "rang de preus"),
                    ],
                    justify="center",
                    style={"padding": "24px 60px 0"},
                ),
            ],
        ),
        html.Hr(style=DIVIDER),
        ##########
        # GENRES #
        ##########
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    [
                        "Una hipòtesi freqüent és que el gènere amb més presència al catàleg sigui també "
                        "el que registra un millor rendiment comercial. En termes d'oferta, el gènere "
                        "predominant és ",
                        html.Strong("Indie", style={"color": ACCENT}),
                        ", amb més de 80.000 títols publicats. ",
                        "Tanmateix, que hi hagi un gran volum d'exemplars d'un gènere en concret no implica necessàriament que aquest sigui el més popular o el millor valorat. "
                        "De fet, quan ordenem per nombre de vendes, el lideratge passa a correspondre a ",
                        html.Strong("Action", style={"color": ACCENT}),
                        ", malgrat disposar inicialment d'un volum d'oferta inferior al d'Indie. ",
                    ],
                    style={**TEXT_FONT},
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_genre_count), md=6),
                        dbc.Col(dcc.Graph(figure=fig_genre_sales), md=6),
                    ]
                ),
            ],
        ),
        html.Hr(style=DIVIDER),
        #########################
        # PRICE vs SATISFACTION #
        #########################
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    [
                        "Per altra banda, un aspecte rellevant que se sol associar amb bona qualitat és el preu. "
                        "Culturalment, les persones associen sovint un preu elevat amb una qualitat superior. "
                        "Per tal de verificar si aquesta associació es manté en el cas dels videojocs de Steam, "
                        "es pot examinar la relació entre el preu i la satisfacció dels usuaris de cadascun dels registres del dataset. "
                        "Si existís tal relació de manera sistemàtica, el núvol de punts mostraria una "
                        "tendència ascendent o descendent clara. ",
                        html.Br(),
                        html.Br(),
                        "Com es pot observar, els punts es distribueixen de manera dispersa i mostren valoracions majoritàriament "
                        "positives en pràcticament tots els nivells de preu. En conseqüència, no s'identifica "
                        "una relació lineal consistent entre el preu i la satisfacció dels usuaris.",
                        html.Br(),
                        html.Br(),
                        "A més, el mateix gràfic incorpora un selector per alternar entre l'escala "
                        "logarítmica i la lineal i facilitar-ne la lectura.",
                    ],
                    style={**TEXT_FONT},
                ),
                dcc.Graph(id="price-scatter", figure=fig_price),
            ],
        ),
        html.Hr(style=DIVIDER),
        #######################
        # PEAK CCU vs SATISFACTION #
        #######################
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    [
                        "Donat que un gran nombre de jugadors actius simultàniament pot ser un indicador de popularitat i visibilitat, "
                        "sembla lògic pensar en intentar plasmar aquesta relació en un gràfic. S'ha considerat que la millor manera de "
                        "fer-ho és mitjançant un núvol de punts on l'eix X representaria el nombre màxim de jugadors actius simultàniament, "
                        "corresponent a la variable Peak CCU i l'eix Y el percentatge de valoracions positives sobre el total, corresponent "
                        "a la variable Satisfaction ratio. A més, el tamany de cada cercle representa el nombre total de recomanacions que ha "
                        "rebut el joc. Com més gran el cercle, més recomanacions ha rebut.",
                        html.Br(),
                        html.Br(),
                        "Aquí observem que hi ha pics alts de nombre de jugadors en diversos jocs però que no tots han presentat una alta satisfacció "
                        "per moltes recomanacions que hagin tingut o nombre de jugadors simultanis. A mesura que augmenta el nombre de jugadors actius, "
                        "sí que es denota que els jugadors recomanen el joc cada cop més, però no sembla que hi hagi una relació directa. S'observen alguns "
                        "casos en què la recomanació i el nombre de jugadors mostren una associació, però també d'altres com ",
                        html.Strong("PUBG", style={"color": ACCENT}),
                        " que té molts jugadors però poca satisfacció i ",
                        html.Strong("DELTARUNE", style={"color": ACCENT}),
                        " que té poca recomanació però molts jugadors satisfets."
                    ],
                    style={**TEXT_FONT},
                ),
                dcc.Graph(id="ccu-scatter", figure=fig_ccu),
            ],
        ),
        html.Hr(style=DIVIDER),
        # ── PROFILE MAP ─────────────────────────────────────────────────────
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    [
                        "Ja per acabar, si s'analitzen conjuntament la satisfacció i la popularitat, és possible classificar els "
                        "jocs en diferents perfils que identifiquen el seu rendiment més enllà de la seva popularitat.",
                        html.Br(),
                        html.Br(),
                        "Aquests perfils són:",
                        html.Br(),
                        "- Blockbuster: jocs que han funcionat molt bé tant en termes de popularitat com de satisfacció.",
                        html.Br(),
                        "- Hidden Gem: jocs que tenen una alta satisfacció però no han arribat a ser molt coneguts.",
                        html.Br(),
                        "- Mainstream: jocs amb un volum de jugadors i valoracions moderades sense destacar especialment.",
                        html.Br(),
                        "- Decebedor: jocs amb gran volum de jugadors però baixa satisfacció.",
                        html.Br(),
                        html.Br(),
                        "De forma clara es visualitzen aquells jocs que han aconseguit destacar respecte la resta i aquells que"
                        ", tot i ser bons, ho podrien haver fet millor.",
                        html.Br(),
                        html.Br(),
                        "Els filtres permeten observar com varia aquesta "
                        "distribució segons el llindar de visibilitat i el volum de ressenyes.",
                    ],
                    style={**TEXT_FONT},
                ),
                dbc.Row(
                    justify="center",
                    style={"marginBottom": "16px"},
                    children=[
                        dbc.Col(
                            [
                                html.Label("Jugadors concurrents mínims:", style={"color": TEXT, "fontSize": "0.9rem"}),
                                dcc.Slider(
                                    id="profile-ccu-min",
                                    min=100,
                                    max=10000,
                                    step=100,
                                    value=1000,
                                    marks={100: "100", 1000: "1K", 5000: "5K", 10000: "10K"},
                                    tooltip={"placement": "bottom"},
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Ressenyes mínimes:", style={"color": TEXT, "fontSize": "0.9rem"}),
                                dcc.Slider(
                                    id="profile-reviews-min",
                                    min=100,
                                    max=5000,
                                    step=100,
                                    value=1000,
                                    marks={100: "100", 1000: "1K", 2500: "2.5K", 5000: "5K"},
                                    tooltip={"placement": "bottom"},
                                ),
                            ],
                            md=4,
                        ),
                    ],
                ),
                dcc.Graph(id="profile-scatter"),
            ],
        ),
        html.Hr(style=DIVIDER),
        # ── CONCLUSIONS ──────────────────────────────────────────────────────
        html.Div(
            style={**SECTION, "textAlign": "center"},
            children=[
                html.H2("Conclusions principals", style={"color": ACCENT, "marginBottom": "32px"}),
                dbc.Row(
                    justify="center",
                    children=[
                        dbc.Col(
                            [
                                conclusion_card(
                                    "Preu i qualitat",
                                    "No s'observa un interval de preu que garanteixi valoracions altes. "
                                    "La recepció dels usuaris es manté heterogènia en tot l'espectre de preus.",
                                ),
                                conclusion_card(
                                    "Pes dels gèneres",
                                    "El gènere Indie concentra el major volum de títols, mentre que Action i RPG "
                                    "destaquen en indicadors vinculats a la implicació dels usuaris i a la recomanació.",
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            [
                                conclusion_card(
                                    "Paradoxa de la popularitat",
                                    "Els jocs amb valors molt elevats de jugadors concurrents no sempre superen el 90% de satisfacció, "
                                    "fet que apunta a una relació imperfecta entre abast massiu i valoració.",
                                ),
                                conclusion_card(
                                    "Títols infrarepresentats",
                                    "L'anàlisi també permet identificar jocs amb una recepció molt positiva però amb una visibilitat comparativament limitada.",
                                ),
                            ],
                            md=5,
                        ),
                    ],
                ),
            ],
        ),
    ],
)

#############
# Callbacks #
#############
@app.callback(
    Output("profile-scatter", "figure"),
    Input("profile-ccu-min", "value"),
    Input("profile-reviews-min", "value"),
)
def update_profile(ccu_min, reviews_min):
    fig, _ = build_profile_scatter(
        steam_df,
        ccu_min=ccu_min,
        reviews_min=reviews_min,
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
