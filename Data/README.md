# Data conventions

`data/raw/` holds immutable copies of downloaded source data. Never edit files here.

`data/interim/` holds reproducible intermediate cleaning and transformation outputs.

`data/processed/` holds final analysis-ready datasets used by notebooks or later models.

Datasets are ignored by Git by default because they may be large or have redistribution restrictions. Preserve source documentation and record each important raw-to-derived transformation in the relevant notebook or methodology note. Keep observed fields, derived variables, and future modelling assumptions clearly distinct.

