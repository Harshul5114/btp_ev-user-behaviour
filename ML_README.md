# Machine Learning Documentation

The authoritative, detailed ML documentation for the project is located at:
**[`docs/ML_README.md`](docs/ML_README.md)**

### Quick Links to ML Models & Notebooks:
* **Overview & Consolidated Metrics:** [`docs/ML_README.md`](docs/ML_README.md)
* **Session Delivered Energy Experiment:**
  - Guide: [`docs/ML_EXPERIMENT.md`](docs/ML_EXPERIMENT.md)
  - Python implementation: [`src/models/session_energy.py`](src/models/session_energy.py)
  - Interactive notebook: [`notebooks/06_ml_session_energy.ipynb`](notebooks/06_ml_session_energy.ipynb)
  - Metric outputs: [`outputs/ml_metrics.csv`](outputs/ml_metrics.csv)
* **Plug-in and Plug-out Times Experiment (Connection Duration & Inter-Session Gaps):**
  - Python implementation: [`src/models/charging_times.py`](src/models/charging_times.py)
  - Interactive notebook: [`notebooks/07_ml_plugin_plugout_times.ipynb`](notebooks/07_ml_plugin_plugout_times.ipynb)
  - Metric outputs: [`outputs/ml_charging_times_metrics.csv`](outputs/ml_charging_times_metrics.csv)
  - Visualizations:
    - Connection Time Actual vs Predicted: [`outputs/ml_connection_time_actual_vs_predicted.png`](outputs/ml_connection_time_actual_vs_predicted.png)
    - Next Gap Actual vs Predicted: [`outputs/ml_next_gap_actual_vs_predicted.png`](outputs/ml_next_gap_actual_vs_predicted.png)
    - Day-of-Week Behavioral Envelopes: [`outputs/ml_charging_times_by_dayofweek.png`](outputs/ml_charging_times_by_dayofweek.png)
