# Masterproef
VLINDER: missing data strategieën voor weer- en klimaatdata.

## Project Setup
### Vereisten
Zorg dat je **Python 3.12** of hoger en **Poetry** geïnstalleerd hebt. Je kan Poetry installeren via:
```bash
pip install poetry
```

### Virtual Environment
Maak een virtuele omgeving aan en installeer de vereisten:
```bash
poetry install
```

Activeer de omgeving met:
```bash
poetry shell
```

### Project uitvoeren
Je kan de verschillende scripts uitvoeren via:
```bash
python <script_naam>.py
```

## Mappenstructuur
```
Masterproef/
│
├── data/                  # Ruwe datasets (niet in GitHub, te groot)
│
├── src/
│   └── scripts/           # Python scripts (preprocessing, training, evaluatie)
│
├── pyproject.toml         # Poetry-configuratie (dependencies, Python-versie)
├── poetry.lock            # Exacte versie-lock van dependencies
└── README.md              # Deze handleiding
```