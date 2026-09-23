# BTP — EV User Behaviour and Grid Impact Modelling

## Project Context

This repository is for a B.Tech Project (BTP) in Electrical Engineering at IIT Delhi.

The broad research goal is to model and eventually predict electric-vehicle (EV) user behaviour and translate that behaviour into spatial-temporal charging demand that can be used for power-system analysis.

The project should eventually connect:

mobility / travel behaviour
        ↓
EV energy consumption and SOC
        ↓
charging behaviour
        ↓
individual EV charging profiles
        ↓
aggregated spatial-temporal EV demand
        ↓
distribution / transmission grid model
        ↓
power-flow and voltage impact analysis

The project is exploratory and the exact final methodology is NOT fixed yet. Keep the codebase modular so individual modelling choices can be replaced later.

---

## Current Immediate Goal

There is an upcoming project presentation.

For now, DO NOT attempt to build the complete system.

The immediate work is:

1. Obtain and inspect travel / household mobility data, initially likely NHTS.
2. Understand the dataset structure and relevant variables.
3. Clean and preprocess only the fields required for EV behaviour modelling.
4. Perform exploratory data analysis.
5. Extract useful behavioural insights and distributions.
6. Reconstruct daily travel schedules / trip chains where possible.
7. Produce presentation-quality plots and preliminary observations.
8. If time permits, build a very simple rule-based conversion from travel behaviour to EV energy demand / charging demand.

The current phase is primarily DATA EXPLORATION, not ML.

Do not start implementing models unless explicitly asked.

---

## Research Questions

The project is interested in questions such as:

### Travel behaviour

- When do users leave home?
- When do they return home?
- When do they arrive at and leave work?
- How many trips are made per day?
- How far does a vehicle travel per trip and per day?
- How long does a vehicle remain parked?
- Where is the vehicle located during different times of day?
- Can users be grouped into recognizable mobility patterns?

### EV behaviour

Given travel behaviour and EV parameters:

- How much battery energy is consumed?
- What is SOC after each trip?
- When is charging required?
- Where is charging likely to occur?
- How much energy does a user need?
- How long would charging take for different charger ratings?
- What does the user's daily charging profile look like?

Important possible EV parameters include:

- battery capacity [kWh]
- energy consumption [kWh/km]
- charger power [kW]
- charging efficiency
- plug-in time
- plug-out / departure time
- SOC at plug-in
- target SOC

These parameters may initially be assumptions or sampled distributions. Keep them separate from observed dataset variables.

### Aggregation

Eventually determine:

- charging demand of one EV
- charging demand of groups of EVs
- charging demand at different locations
- total EV demand versus time
- differences between home/work/other charging
- variability caused by stochastic user behaviour

### Power-system impact

Eventually map charging demand to buses in a test power network.

Questions include:

- Which locations/buses experience high EV concentration?
- How does EV charging change the bus load?
- How does it affect voltage magnitude?
- How does it affect system losses?
- At what locations/times may grid support be required?
- How much additional generation/load support is required?

A no-EV power-flow case should eventually serve as the baseline.

Possible initial test networks include standard IEEE bus systems.

---

## Long-Term ML Direction

ML is NOT the current task, but the repository should allow it later.

Potential future tasks include:

- predicting next-day user travel schedules
- predicting plug-in / plug-out times
- predicting charging probability
- predicting charging location
- predicting daily energy demand
- clustering users by behaviour
- learning recurring mobility patterns
- comparing rule-based, probabilistic and ML approaches

Do not prematurely choose a particular ML model.

Start with understanding the data and establishing meaningful baselines.

---

## Data Sources

Likely datasets include:

### NHTS

National Household Travel Survey data is expected to be an important source.

Useful information may include:

- household ID
- person ID
- vehicle ID
- trip ID
- trip start/end time
- trip distance
- origin/destination
- trip purpose
- vehicle information

Do NOT assume exact column names.

Always inspect the actual downloaded dataset and document the mapping from raw columns to project variables.

### Household electricity data

Household electricity-consumption datasets may later be used to obtain a non-EV/base-load profile.

This is separate from the EV mobility dataset.

### Synthetic data

Synthetic behaviour may eventually be generated if real datasets lack required EV-specific parameters.

Clearly distinguish:

1. observed data,
2. derived variables,
3. assumed parameters,
4. synthetic/generated data.

Never silently treat assumptions as observations.

---

## Modelling Philosophy

Prefer a transparent pipeline.

For example:

trip data
→ daily vehicle schedules
→ travel distance
→ energy consumption
→ SOC trajectory
→ charging decision
→ charging session
→ charging power time series
→ aggregate EV load

Avoid building a monolithic simulation.

Each stage should have a clean interface so assumptions can be replaced later.

---

## Suggested Repository Architecture

The exact structure may evolve, but use approximately:

.
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_travel_eda.ipynb
│   ├── 03_trip_chains.ipynb
│   └── 04_ev_load_prototype.ipynb
│
├── src/
│   ├── data/
│   ├── behaviour/
│   ├── charging/
│   ├── grid/
│   ├── models/
│   ├── visualization/
│   └── utils/
│
├── configs/
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── presentation/
│
├── tests/
│
└── docs/
    ├── methodology/
    ├── literature/
    └── notes/

Do not create unnecessary abstraction or boilerplate just because these directories exist.

Empty directories can contain `.gitkeep` files if required.

---

## Directory Intent

### data/raw

Original downloaded datasets.

Never modify these files.

Large datasets should generally not be committed to Git.

### data/interim

Intermediate cleaned or transformed data.

### data/processed

Final datasets used by modelling/analysis.

### notebooks

Exploration and research.

During the current presentation phase, most work will happen here.

Notebooks should have a clear numerical order.

### src/data

Reusable dataset loading, cleaning and preprocessing.

### src/behaviour

Travel schedules, trip chains and user-behaviour representations.

### src/charging

SOC, energy consumption and charging simulation.

### src/grid

Power-flow models and mapping charging demand onto power-system buses.

### src/models

Future statistical / ML prediction models.

Do not implement these until required.

### src/visualization

Reusable plotting functions.

### configs

Model/simulation parameters that should not be hardcoded throughout notebooks.

### outputs

Generated figures, tables and presentation artifacts.

### docs/literature

Notes from research papers.

### docs/methodology

Descriptions of modelling decisions and assumptions.

---

## Coding Principles

Use Python for the main analysis.

Prefer:

- pandas
- numpy
- matplotlib
- scipy / scikit-learn when actually needed
- appropriate power-system libraries later if required

Do not add dependencies until they are needed.

Keep notebooks useful for exploration but move reusable logic into `src/` once it becomes stable.

Avoid giant notebooks containing the entire project.

Use functions when transformations become repeated.

Prefer readable research code over excessive software-engineering abstraction.

---

## Reproducibility

Where randomness is used:

- expose the random seed
- document distributions
- distinguish sampled values from measured values

Keep modelling assumptions configurable rather than scattered as magic numbers.

Example future configuration:

battery_capacity_kwh
energy_consumption_kwh_per_km
home_charger_kw
work_charger_kw
fast_charger_kw
charging_efficiency
minimum_soc
target_soc

Do not create these values yet unless requested.

---

## Data Analysis Rules

When examining a new dataset:

1. Inspect shape and columns.
2. Identify ID columns and observational unit.
3. Inspect missing values.
4. Inspect units.
5. Check categorical encodings.
6. Check impossible/outlier values.
7. Determine whether rows represent households, people, vehicles, or trips.
8. Determine how tables can be linked.
9. Document every important transformation.

Never guess what an encoded column means. Consult the dataset documentation/codebook.

Before dropping data, explain why.

---

## Presentation Phase

The immediate presentation should demonstrate progress rather than pretend the final model is complete.

Useful outputs include:

- dataset overview
- methodology/pipeline diagram
- trip-start distribution
- trip-arrival distribution
- trip-distance distribution
- trips per vehicle per day
- daily vehicle distance
- home/work/other location patterns
- parking-duration distributions
- example daily trip chains
- preliminary implications for EV charging

Plots should be readable and suitable for presentation slides.

Every plot should have:

- meaningful title
- labelled axes
- units
- sensible time formatting
- short interpretation

Do not produce dozens of plots without a research question.

---

## Research References / Conceptual Direction

The literature being studied includes work on:

- EV load modelling for power-flow analysis
- spatial-temporal EV charging simulation
- aggregated EV charging behaviour
- probabilistic EV driver charging behaviour
- quasi-steady-state EV load modelling

A particularly relevant conceptual approach models:

travel behaviour
→ transportation trajectory
→ energy consumption
→ charging decision
→ charging location/time
→ aggregated bus charging load

Our project does NOT need to reproduce any one paper exactly.

The papers should guide modelling choices while the implementation remains suitable for the available datasets and BTP scope.

---

## Current Priority Order

Unless explicitly instructed otherwise:

1. Dataset acquisition / inspection
2. Data cleaning
3. Exploratory analysis
4. Travel-pattern extraction
5. Simple EV energy model
6. Simple charging model
7. Aggregate charging load
8. Grid integration
9. Prediction / ML

The current work is around stages 1–4.

---

## Instructions for Codex

Do not automatically implement the entire roadmap.

When given a task:

1. Work only on the requested stage.
2. Inspect existing files before creating replacements.
3. Prefer incremental changes.
4. Explain major assumptions.
5. Do not invent dataset schemas.
6. Do not introduce ML merely because this is an ML-oriented project.
7. Keep future extensibility in mind without overengineering.
8. Preserve raw data.
9. Keep generated outputs separate from source code.
10. Update documentation when an important modelling decision is made.

If requirements are ambiguous and the choice could materially affect the research methodology, ask before committing to it.