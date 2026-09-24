"""Deterministic, single-day EV charging prototype on observed NHTS trip chains.

The mobility inputs are reported/derived NHTS observations. ``EVParameters`` are
synthetic assumptions; no battery or charger specifications are inferred from NHTS.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


MILES_TO_KM = 1.609344
DAY_START_MIN = 240       # NHTS survey day begins at 04:00.
DAY_END_MIN = 1680        # Survey day ends at next-day 04:00.
BIN_MIN = 15
BIN_STARTS_MIN = np.arange(DAY_START_MIN, DAY_END_MIN, BIN_MIN)
SOC_TIMES_MIN = np.arange(DAY_START_MIN, DAY_END_MIN + BIN_MIN, BIN_MIN)
_TOL = 1e-8


@dataclass(frozen=True)
class EVParameters:
    """Synthetic, configurable EV and charging assumptions."""

    battery_capacity_kwh: float = 60.0
    energy_consumption_kwh_per_km: float = 0.18
    home_charger_grid_kw: float = 7.4
    charging_efficiency: float = 0.90
    initial_soc: float = 0.80
    target_soc: float = 0.90

    def __post_init__(self) -> None:
        if self.battery_capacity_kwh <= 0 or self.energy_consumption_kwh_per_km <= 0:
            raise ValueError("Battery capacity and energy consumption must be positive")
        if self.home_charger_grid_kw <= 0 or not 0 < self.charging_efficiency <= 1:
            raise ValueError("Charger power must be positive; efficiency must be in (0, 1]")
        if not 0 <= self.initial_soc <= 1 or not 0 <= self.target_soc <= 1:
            raise ValueError("Initial and target SoC must be fractions in [0, 1]")


@dataclass
class SimulationResult:
    vehicle_id: str
    events: pd.DataFrame
    power_kw: np.ndarray              # 96 average grid-side powers, one per 15-minute bin.
    soc_fraction: np.ndarray          # 97 boundary states from 04:00 to next 04:00.
    feasible: bool
    failure_reason: str | None = None

    @property
    def grid_energy_kwh(self) -> float:
        return float(self.events.grid_charge_kwh.sum())

    @property
    def driving_energy_kwh(self) -> float:
        return float(self.events.driving_kwh.sum())


def _validate_chain(legs: pd.DataFrame, gaps: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    if legs.empty or legs.VEHCASEID.nunique() != 1:
        raise ValueError("Pass the legs of exactly one household vehicle")
    vehicle_id = str(legs.VEHCASEID.iloc[0])
    legs = legs.sort_values("vehicle_leg_index").reset_index(drop=True)
    gaps = gaps.copy()
    if len(legs) < 2 or not legs.chain_status.eq("complete").all():
        raise ValueError("Simulation requires a multi-leg, purpose-continuous chain")
    if len(gaps) != len(legs) - 1 or gaps.TDCASEID.duplicated().any():
        raise ValueError("Expected one unique internal gap after each non-final leg")
    if not gaps.VEHCASEID.eq(vehicle_id).all() or not gaps.analysis_eligible.eq(True).all():
        raise ValueError("Use only eligible gaps for the same vehicle")
    if not np.array_equal(legs.vehicle_leg_index.to_numpy(), np.arange(1, len(legs) + 1)):
        raise ValueError("Vehicle leg indices must be consecutive from one")
    if legs.survey_start_min.min() < DAY_START_MIN or legs.survey_end_min.max() > DAY_END_MIN:
        raise ValueError("All legs must fit inside the observed 04:00–04:00 survey day")
    gap_by_arrival = gaps.set_index("TDCASEID", verify_integrity=True)
    for i in range(len(legs) - 1):
        here, nxt = legs.iloc[i], legs.iloc[i + 1]
        if here.TDCASEID not in gap_by_arrival.index:
            raise ValueError("Missing gap after a non-final leg")
        gap = gap_by_arrival.loc[here.TDCASEID]
        if (str(gap.next_TDCASEID) != str(nxt.TDCASEID)
                or abs(float(gap.parking_start_min) - float(here.survey_end_min)) > _TOL
                or abs(float(gap.parking_end_min) - float(nxt.survey_start_min)) > _TOL):
            raise ValueError("Gap endpoints or next-trip ID disagree with trip chain")
    return vehicle_id, legs, gap_by_arrival


def _power_and_soc(events: pd.DataFrame, params: EVParameters) -> tuple[np.ndarray, np.ndarray]:
    """Use exact interval overlap for bin-average power and a linear trip display path."""
    power = np.zeros(len(BIN_STARTS_MIN), dtype=float)
    battery = np.full(len(SOC_TIMES_MIN),
                      params.battery_capacity_kwh * params.initial_soc, dtype=float)
    for event in events.itertuples(index=False):
        if event.event_type == "trip":
            duration = event.end_min - event.start_min
            if duration > 0:
                fraction = np.clip((SOC_TIMES_MIN - event.start_min) / duration, 0, 1)
            else:
                fraction = (SOC_TIMES_MIN >= event.end_min).astype(float)
            battery -= event.driving_kwh * fraction
        elif event.grid_charge_kwh > 0:
            start, end = event.start_min, event.charge_end_min
            overlap = np.maximum(0, np.minimum(BIN_STARTS_MIN + BIN_MIN, end)
                                 - np.maximum(BIN_STARTS_MIN, start))
            power += params.home_charger_grid_kw * overlap / BIN_MIN
            charging_minutes = np.clip(SOC_TIMES_MIN - start, 0, end - start)
            battery += (params.charging_efficiency * params.home_charger_grid_kw
                        * charging_minutes / 60)
    return power, battery / params.battery_capacity_kwh


def simulate_vehicle(legs: pd.DataFrame, gaps: pd.DataFrame,
                     params: EVParameters) -> SimulationResult:
    """Drive each leg, then charge immediately during eligible internal Home gaps.

    A trip requiring more energy than is available returns ``feasible=False`` with
    a flagged event. It does not silently clip the battery to zero. No pre-first
    or post-final parking is synthesized.
    """
    vehicle_id, legs, gap_by_arrival = _validate_chain(legs, gaps)
    battery = params.battery_capacity_kwh * params.initial_soc
    target = params.battery_capacity_kwh * params.target_soc
    events: list[dict] = []

    for i, leg in legs.iterrows():
        start, end = float(leg.survey_start_min), float(leg.survey_end_min)
        miles = float(leg.trip_distance_miles)
        if not np.isfinite([start, end, miles]).all() or miles < 0 or end < start:
            raise ValueError(f"Invalid trip geometry for {leg.TDCASEID}")
        distance_km = miles * MILES_TO_KM
        required = distance_km * params.energy_consumption_kwh_per_km
        before = battery
        if (end == start and distance_km > _TOL) or required > before + _TOL:
            reason = ("positive distance with zero travel time" if end == start
                      else "trip energy exceeds available battery energy")
            events.append(dict(event_type="trip_infeasible", vehicle_id=vehicle_id,
                               trip_id=str(leg.TDCASEID), start_min=start, end_min=end,
                               from_purpose=leg.origin_group, to_purpose=leg.destination_group,
                               location="travel", distance_miles=miles,
                               distance_km=distance_km, driving_kwh=required,
                               grid_charge_kwh=0.0, battery_charge_kwh=0.0,
                               battery_before_kwh=before, battery_after_kwh=np.nan,
                               soc_before=before / params.battery_capacity_kwh,
                               soc_after=np.nan, charge_end_min=np.nan))
            return SimulationResult(vehicle_id, pd.DataFrame(events),
                                    np.full(len(BIN_STARTS_MIN), np.nan),
                                    np.full(len(SOC_TIMES_MIN), np.nan), False,
                                    f"{leg.TDCASEID}: {reason}")
        battery = before - required
        events.append(dict(event_type="trip", vehicle_id=vehicle_id,
                           trip_id=str(leg.TDCASEID), start_min=start, end_min=end,
                           from_purpose=leg.origin_group, to_purpose=leg.destination_group,
                           location="travel", distance_miles=miles,
                           distance_km=distance_km, driving_kwh=required,
                           grid_charge_kwh=0.0, battery_charge_kwh=0.0,
                           battery_before_kwh=before, battery_after_kwh=battery,
                           soc_before=before / params.battery_capacity_kwh,
                           soc_after=battery / params.battery_capacity_kwh,
                           charge_end_min=np.nan))

        if i == len(legs) - 1:
            continue
        gap = gap_by_arrival.loc[leg.TDCASEID]
        park_start, park_end = float(gap.parking_start_min), float(gap.parking_end_min)
        before = battery
        charging_minutes = 0.0
        if gap.destination_group == "Home" and battery < target - _TOL:
            needed_grid_kwh = (target - battery) / params.charging_efficiency
            charging_minutes = min(park_end - park_start,
                                   60 * needed_grid_kwh / params.home_charger_grid_kw)
        grid_kwh = params.home_charger_grid_kw * charging_minutes / 60
        stored_kwh = params.charging_efficiency * grid_kwh
        battery += stored_kwh
        if ((grid_kwh > 0 and battery > target + _TOL)
                or battery > params.battery_capacity_kwh + _TOL):
            raise AssertionError("Charging overshot the target or battery capacity")
        events.append(dict(event_type="parking", vehicle_id=vehicle_id,
                           trip_id=str(leg.TDCASEID), start_min=park_start,
                           end_min=park_end, from_purpose=gap.destination_group,
                           to_purpose=gap.destination_group,
                           location=gap.destination_group, distance_miles=0.0,
                           distance_km=0.0, driving_kwh=0.0,
                           grid_charge_kwh=grid_kwh, battery_charge_kwh=stored_kwh,
                           battery_before_kwh=before, battery_after_kwh=battery,
                           soc_before=before / params.battery_capacity_kwh,
                           soc_after=battery / params.battery_capacity_kwh,
                           charge_end_min=(park_start + charging_minutes
                                           if grid_kwh > 0 else np.nan)))

    table = pd.DataFrame(events)
    power, soc = _power_and_soc(table, params)
    if not np.isclose(power.sum() * BIN_MIN / 60, table.grid_charge_kwh.sum(), atol=_TOL):
        raise AssertionError("15-minute profile energy disagrees with charged grid energy")
    if not np.isclose(soc[-1] * params.battery_capacity_kwh,
                      params.battery_capacity_kwh * params.initial_soc
                      - table.driving_kwh.sum() + table.battery_charge_kwh.sum(), atol=_TOL):
        raise AssertionError("Vehicle-day battery energy is not conserved")
    if soc.min() < -_TOL or soc.max() > 1 + _TOL:
        raise AssertionError("Simulated SoC left physical [0, 1] bounds")
    return SimulationResult(vehicle_id, table, power, soc, True)
