"""Known-answer synthetic checks for terminal parking; no measured data used."""
import unittest

import numpy as np
import pandas as pd

from src.charging.simple_ev import (
    EVParameters, SimulationResult, reconstruct_terminal_home_charge,
)


class TerminalChargingTests(unittest.TestCase):
    def fixture(self, arrival, departure, final_soc, destination="Home"):
        legs = pd.DataFrame({"vehicle_leg_index": [1, 2],
                             "survey_start_min": [departure, arrival - 10],
                             "survey_end_min": [departure + 10, arrival],
                             "destination_group": ["Other", destination]})
        result = SimulationResult("synthetic", pd.DataFrame(), np.zeros(96),
                                  np.full(97, final_soc), True)
        return legs, result

    def test_charge_crossing_survey_boundary_preserves_energy(self):
        params = EVParameters(home_charger_grid_kw=6, charging_efficiency=1)
        # 03:30 next day to 04:30: 3 kWh on each side of the 04:00 boundary.
        legs, baseline = self.fixture(1650, 480, .8)
        result = reconstruct_terminal_home_charge(legs, baseline, params)
        self.assertAlmostEqual(result.grid_energy_kwh, 6)
        np.testing.assert_allclose(result.power_kw[[94, 95, 0, 1]], 6)
        self.assertEqual(np.count_nonzero(result.power_kw), 4)
        self.assertAlmostEqual(result.charge_end_min, 1710)

    def test_departure_limits_charge_and_efficiency_is_applied(self):
        params = EVParameters(home_charger_grid_kw=6, charging_efficiency=.9)
        legs, baseline = self.fixture(1650, 250, .5)
        result = reconstruct_terminal_home_charge(legs, baseline, params)
        self.assertAlmostEqual(result.departure_min, 1690)
        self.assertAlmostEqual(result.charge_end_min, 1690)
        self.assertAlmostEqual(result.grid_energy_kwh, 4)
        self.assertAlmostEqual(result.grid_energy_kwh * params.charging_efficiency, 3.6)

    def test_non_home_terminal_adds_no_energy(self):
        legs, baseline = self.fixture(1080, 480, .5, "Work")
        result = reconstruct_terminal_home_charge(legs, baseline, EVParameters())
        self.assertEqual(result.grid_energy_kwh, 0)
        self.assertIsNone(result.arrival_min)

    def test_target_already_met(self):
        legs, baseline = self.fixture(1080, 480, .9)
        result = reconstruct_terminal_home_charge(legs, baseline, EVParameters())
        self.assertEqual(result.grid_energy_kwh, 0)


if __name__ == "__main__":
    unittest.main()
