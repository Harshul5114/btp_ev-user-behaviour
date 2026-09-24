# Deterministic EV SoC and Home charging prototype

Implementation: `src/charging/simple_ev.py`; reproducible analysis and figures: `notebooks/05_ev_load_prototype.ipynb`.

## Input and selection

The simulator reads the **interim** `vehicle_trip_chains.csv` and `vehicle_parking_intervals.csv` created by notebook 03. Their fields are defined in [vehicle_trip_chain_schema.md](vehicle_trip_chain_schema.md). A day enters the prototype if it has at least two candidate driver legs, complete reported-purpose continuity, all trips within the NHTS 04:00–04:00 survey window, and one eligible internal gap after each nonfinal leg. Gap keys and endpoints are checked against the trip sequence. The input is person-reported candidate vehicle movement, not verified GPS continuity. Trip weights are not applied to these derived vehicle-day charging events.

## Synthetic EV and rule

The baseline assigns **one hypothetical EV to every selected NHTS vehicle-day**: 60 kWh battery, 0.18 kWh/km driving consumption, 80% initial SoC at 04:00, 7.4 kW grid-side Home charger, 90% grid-to-battery efficiency, and 90% target SoC. All six values are configurable assumptions, **not NHTS observations**. Miles are converted with 1.609344 km/mile.

Each trip deducts `distance_km × consumption_kwh_per_km` from battery energy. If a trip requires more energy than remains, that vehicle-day is flagged infeasible and has no demand curve in the sample aggregate; SoC is never silently clipped. A positive-distance zero-time trip is also flagged.

In an observed internal gap labelled Home, charging starts immediately whenever SoC is below target. Grid draw stays at 7.4 kW until either the next observed departure or target SoC. Battery gain is `grid_kWh × efficiency`. Other destinations do not charge. No pre-first or post-final parking time is invented, including after a final Home arrival.

The 15-minute `P_i(t)` is **grid-side bin-average power**, calculated from exact overlap between charging intervals and each bin; a partially used final bin can be below 7.4 kW. For the displayed `SOC_i(t)`, trip energy is spread uniformly across its reported travel duration; event-level before/after energies use the full trip energy. The simulator checks battery-energy balance, SoC bounds, departure and target stops, and equality between event charging energy and the integrated power profile.

The aggregate is a direct, **unweighted sum of feasible sample vehicle curves** over the NHTS survey day. It is neither a US EV-demand forecast nor a grid-load scenario. The missing post-final Home interval especially suppresses evening and overnight charging. Norway's residential charging reports are cited only as a separate empirical sanity check; their inferred user power and SoC fields are not treated as measurements.
