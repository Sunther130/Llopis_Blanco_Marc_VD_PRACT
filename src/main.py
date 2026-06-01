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
    build_genre_count_treemap,
    build_genre_sales_treemap,
    build_price_scatter,
    build_profile_scatter,
    build_retention_scatter,
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
fig_genre_count = build_genre_count_treemap(steam_df, genre_cols)
fig_genre_sales = build_genre_sales_treemap(steam_df, genre_cols)
fig_price = build_price_scatter(steam_df, genre_cols)
fig_retention = build_retention_scatter(steam_df)

n_games = f"{steam_df.shape[0]:,}"
year_min = int(steam_df["Release_year"].min())
year_max = int(steam_df["Release_year"].max())
n_genres = len(genre_cols)
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
                        "Entenen per satisfacció el percentatge de ressenyes positives que ha rebut cada joc respecte del total.",
                        html.Br(),
                        html.Br(),
                        "Si existís tal relació de manera sistemàtica, el núvol de punts mostraria una "
                        "tendència ascendent o descendent clara. ",
                        html.Br(),
                        html.Br(),
                        "Com es pot observar, els punts es distribueixen de manera dispersa i mostren valoracions majoritàriament "
                        "positives en pràcticament tots els nivells de preu. En conseqüència, no s'identifica "
                        "una relació lineal consistent entre el preu i la satisfacció dels usuaris.",
                        html.Br(),
                        html.Br(),
                        "El que sí apareix és una concentració en els valors més baixos. Tot i que recordem que el gènere Indie "
                        "és el més abundant, i aquest sol tenir un preu relativament baix. Per tant, és normal veure que hi ha molts punts "
                        "en aquesta zona.",
                    ],
                    style={**TEXT_FONT},
                ),
                dcc.Graph(id="price-scatter", figure=fig_price),
            ],
        ),
        html.Hr(style=DIVIDER),
        ###############
        # PROFILE MAP #
        ###############
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    [
                        "Si el preu no explica la satisfacció, potser ho fa la popularitat. Un gran nombre de jugadors actius simultàniament "
                        "és un dels indicadors de visibilitat més directes de la plataforma. Tanmateix, tampoc aquí s'observa una relació simple. "
                        "Hi ha títols amb pics altíssims de jugadors que no han obtingut una satisfacció especialment alta, i viceversa.",
                        html.Br(),
                        html.Br(),
                        "El que sí permet aquest eix és construir perfils d'acollida que categoritzen els videojocs segons la seva posició "
                        "en l'espai, distingint entre: ",
                        html.Ul(
                            [
                                html.Li([html.Strong("Decebedor: "), "jocs amb baix volum de jugadors i baixa satisfacció."]),
                                html.Li([html.Strong("Mainstream: "), "jocs amb un volum de jugadors i valoracions moderades sense destacar especialment."]),
                                html.Li([html.Strong("Blockbuster: "), "jocs que han funcionat molt bé tant en termes de popularitat com de satisfacció."]),
                                html.Li([html.Strong("Hidden Gem: "), "jocs que tenen una alta satisfacció però no han arribat a ser molt coneguts."]),
                                html.Li([html.Strong("Polaritzat: "), "jocs amb gran volum de jugadors però baixa satisfacció."]),
                            ],
                            style={"marginTop": "8px", "paddingLeft": "20px"},
                        ),
                        html.Br(),
                        "De forma clara es visualitzen aquells jocs que han aconseguit destacar respecte la resta i aquells que, "
                        "tot i ser bons, ho podrien haver fet millor.",
                        html.Br(),
                        html.Br(),
                        "Exploreu com canvia el mapa ajustant els llindars de jugadors i ressenyes.",
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
        #########################
        # RETENTION & INTENSITY #
        #########################
        html.Div(
            style=SECTION,
            children=[
                html.P(
                    [
                        "Fins i tot, podem anar més enllà i analitzar la retenció dels jugadors. "
                        "Quants segueixen jugant activament un joc després d'haver-hi invertit temps? Comparant "
                        "el temps de joc recent (dues setmanes) amb el temps total acumulat, aconseguim generar un "
                        "rati de retenció.",
                        html.Br(),
                        html.Br(),
                        "Resultant en una taula de dispersió que forma 4 quadrants delimitats per una línia discontínua "
                        "segons la seva intensitat d'ús i la seva capacitat de retenció: ",
                        html.Ul(
                            [
                                html.Li(
                                    [html.Strong("Quadrant superior dret: "),
                                     "jocs amb alta retenció i alta intensitat d'ús, indicant títols que mantenen els "
                                     "jugadors compromesos a llarg termini."]),
                                html.Li(
                                    [html.Strong("Quadrant superior esquerre: "),
                                     "jocs amb alta retenció però baixa intensitat d'ús, indicant títols que mantenen "
                                     "els jugadors compromesos però amb sessions de joc més curtes."]),
                                html.Li(
                                    [html.Strong("Quadrant inferior dret: "),
                                     "jocs amb baixa retenció però alta intensitat d'ús, indicant títols que atrauen jugadors "
                                     "intensament però no els mantenen a llarg termini."]),
                                html.Li(
                                    [html.Strong("Quadrant inferior esquerre: "),
                                     "jocs amb baixa retenció i baixa intensitat d'ús, indicant títols que no aconsegueixen "
                                     "mantenir els jugadors compromesos."]),
                            ],
                            style={"marginTop": "8px", "paddingLeft": "20px"},
                        ),
                        html.Br(),
                        "A més, el color codifica l'asimetria entre la mitjana i la mediana del temps de joc on cada número en l'escala és quantes vegades "
                        "la mitjana supera la mediana. "
                        "Valors alts (blau fosc) indiquen que uns pocs jugadors juguen molt més que la "
                        "majoria, mentre que valors baixos (blau clar) suggereixen un ús més distribuït. D'aquesta manera, els foscos "
                        "poden indicar títols que, tot i tenir una alta intensitat d'ús, depenen d'una base de jugadors molt reduïda que hi "
                        "inverteix molt de temps, mentre que els clars poden suggerir títols amb un ús més generalitzat entre la seva "
                        "comunitat de jugadors.",
                    ],
                    style={**TEXT_FONT},
                ),
                dcc.Graph(figure=fig_retention),
            ],
        ),
        html.Hr(style=DIVIDER),
        ###############
        # CONCLUSIONS #
        ###############
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
                                    "El gènere Indie concentra el major volum de títols publicats, mentre que Action "
                                    "s'imposa com el gènere líder en vendes estimades, malgrat tenir un volum "
                                    "d'oferta inferior al d'Indie.",
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
                                    "Retenció i intensitat d'ús",
                                    "La majoria de títols s'agrupen en el quadrant inferior dret, amb baixa retenció recent "
                                    "i un ús total moderat. Només un conjunt reduït de jocs aconsegueix combinar una alta "
                                    "intensitat acumulada amb un rati de retenció elevat.",
                                ),
                            ],
                            md=5,
                        ),
                    ],
                ),
                dbc.Row(
                    justify="center",
                    style={"marginTop": "32px"},
                    children=[
                        dbc.Col(
                            html.P(
                                [
                                    "En conjunt, cap factor aïllat determina l'èxit d'un joc a Steam. "
                                    "El gènere condiciona el volum d'oferta i de vendes, però no garanteix qualitat. "
                                    "El preu és irrellevant com a predictor de satisfacció. "
                                    "La popularitat pot coexistir amb valoracions baixes. "
                                    "I la retenció, la capacitat de mantenir els jugadors compromesos al llarg del temps, "
                                    "és potser l'indicador més honest de si un joc ha aconseguit realment el seu objectiu.",
                                ],
                                style={**TEXT_FONT, "textAlign": "center", "fontStyle": "italic"},
                            ),
                            md=8,
                        )
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
