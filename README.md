# BTP — EV User Behaviour and Grid Impact Modelling

This B.Tech Project (Electrical Engineering, IIT Delhi) investigates how EV travel behaviour can be translated into spatial-temporal charging demand and, later, assessed for its effect on electricity networks.

## Current focus

The immediate objective is a presentation-ready data-exploration phase: acquire and inspect mobility data, document its structure, clean only necessary fields, explore travel behaviour, and reconstruct daily trip chains where possible. This repository does not yet implement EV charging, machine-learning, or power-flow models.

## Intended pipeline

`mobility data → travel schedules/trip chains → EV energy and SOC → charging sessions → aggregated charging load → grid impact analysis`

Only the mobility-data stages are currently in scope. Later stages are deliberately separated so their assumptions can be replaced independently.

## Repository layout

- `data/` — managed raw, interim, and processed datasets; see `data/README.md`.
- `notebooks/` — numbered exploratory notebooks for the presentation phase.
- `src/` — reusable Python modules, organised by pipeline stage.
- `configs/` — future parameter/configuration files.
- `outputs/` — generated figures, tables, and presentation artefacts (not versioned).
- `docs/` — methodology, literature, and working notes.
- `tests/` — future tests for reusable code.

The two supplied dataset folders are kept under `data/raw/` as source material. Future imports should be placed there without altering their originals.

## Setup

Create and activate a virtual environment, then install the exploration dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open the first notebook when data inspection is explicitly begun:

```powershell
jupyter notebook notebooks/01_data_inspection.ipynb
```
