# M2.959 - Visualització de dades
## PRÀCTICA - Projecte de visualització

Alumne/a: Marc Llopis I Blanco

Professor/a responsable: Julià Minguillón Alfonso

Crèdits: 6

2026-06-05

---


## Organització del projecte

El projecte es divideix en les següents parts:

```text
├── LICENSE.md                # Llicència del projecte
├── README.md                 # Documentació general i descripció de l'estructura
├── data/
│   ├── steam_dataset.csv     # Dataset original amb les dades de Steam
│   └── steam_clean.parquet   # Dataset net i preparat
└── src/
	├── __init__.py           # Marca el directori com a paquet Python
	├── charts.py             # Funcions que construeixen els gràfics Plotly
	├── clean.py              # Neteja, transformació i enriquiment del dataset
	├── main.py               # Punt d'entrada de l'aplicació Dash
	└── settings.py           # Constants visuals, textos i rutes globals
```


## Preparació

Per a poder executar el notebook de la pràctica, és necessari seguir els següents passos per a configurar l'entorn de treball i instal·lar les dependències necessàries:

1. Crear i activar un entorn virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux / macOS
   .venv/Scripts/activate           # Windows
   ```

2. Instal·lar dependències:
   ```bash
   pip install -r requirements/requirements.txt
   ```

3. Iniciar l'aplicació Dash:
   ```bash
   python src/main.py
   ```

4. Accedir a l'aplicació a través del navegador web:
   ```
   http://localhost:8050
   ```

5. Per a aturar l'aplicació, utilitzar `Ctrl + C` al terminal on s'està executant.

6. Per sortir de l'entorn virtual:
   ```bash
   deactivate
   ```


## Referències

- Antoine Pitrou (2026) *pathlib* (Versió 3.4.0) https://docs.python.org/3/library/pathlib.html
- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. *Array programming with NumPy*. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
- Hunter, J.D. (2025) *Matplotlib* (Versió 3.10.0) https://matplotlib.org/
- McKinney W. (2026) *Pandas* (Versió 3.0.3) https://pandas.pydata.org/docs/index.html
- Python Software Foundation (2026). *The Python Standard Library*. https://docs.python.org/3/library/
- Plotly Technologies Inc. (2026). *Dash*. https://dash.plotly.com/
