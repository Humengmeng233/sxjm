"""Run all four C-problem models and create tables, workbooks, and figures.

Pipeline:
1. Data input: read normalized CSV files and supplied result templates.
2. Parameter initialization: validate physical, risk, and gate parameters.
3. Model calls: solve Q1, Q2, Q3, Q4-2, and Q4-3.
4. Result output: write CSV, XLSX, PNG, JSON, and Markdown outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, time
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import openpyxl
import pandas as pd

from .forecasting import causal_profile_forecast, intraday_net_forecast
from .model_config import ModelConfig
from .optimization import (
    PlanResult,
    adjustment_cost,
    scenario_future_costs,
    simulate_causal_dispatch,
    solve_energy_plan,
)
from .visualization import (
    plot_monthly_costs,
    plot_question1,
    plot_question2_forecast,
    plot_strategy_comparison,
)


RELEASE_SLOTS = (0, 36, 72, 108)
RELEASE_LABELS = ("00:00", "06:00", "12:00", "18:00")


@dataclass
class DataBundle:
    typical: pd.DataFrame
    dates: pd.DatetimeIndex
    load_kwh: np.ndarray
    pv_kwh: np.ndarray
    net_kwh: np.ndarray
    variable_price: np.ndarray
    fixed_price: np.ndarray
    pv_forecast_kwh: np.ndarray


@dataclass
class ModeResult:
    name: str
    dates: pd.DatetimeIndex
    original_plan_kwh: np.ndarray
    accepted_plan_kwh: np.ndarray
    used_plan_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    storage_kwh: np.ndarray
    storage_start_kwh: np.ndarray
    forecast_center_kwh: np.ndarray
    forecast_risk_kwh: np.ndarray
    price_yuan_per_kwh: np.ndarray
    daily: pd.DataFrame
    gates: pd.DataFrame


def _matrix_from_long(frame: pd.DataFrame, column: str) -> np.ndarray:
    ordered = frame.sort_values(["operating_date", "interval_index"])
    dates = ordered["operating_date"].nunique()
    if len(ordered) != dates * 144:
        raise ValueError(f"{column} 不是完整的 日期×144 时段矩阵")
    return ordered[column].to_numpy(dtype=float).reshape(dates, 144)


def load_inputs(root: Path, config: ModelConfig) -> DataBundle:
    """Step 1: read processed inputs and enforce shape/time checks."""

    processed = root / "data" / "processed"
    typical = pd.read_csv(processed / "typical_day_10min.csv")
    actual = pd.read_csv(
        processed / "year_actuals_tariff_10min.csv",
        parse_dates=["operating_date", "interval_start", "interval_end"],
    )
    forecast_long = pd.read_csv(
        processed / "pv_forecasts_10min_linear.csv",
        parse_dates=["issue_date", "issue_timestamp", "target_timestamp"],
    )

    if len(typical) != 144:
        raise ValueError("典型日数据必须正好包含 144 个10分钟时段")
    if actual["operating_date"].nunique() != 365:
        raise ValueError("年度实际数据必须包含 365 个自然日")
    if actual.duplicated(["operating_date", "interval_index"]).any():
        raise ValueError("年度数据存在重复 日期×时段 键")

    actual = actual.sort_values(["operating_date", "interval_index"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(actual["operating_date"].drop_duplicates().sort_values())
    load = _matrix_from_long(actual, "load_energy_kwh")
    pv = _matrix_from_long(actual, "pv_actual_energy_kwh")
    variable_price = _matrix_from_long(actual, "electricity_price_yuan_per_kwh")
    fixed_price = typical["electricity_price_yuan_per_kwh"].to_numpy(dtype=float)

    date_to_index = {date.normalize(): idx for idx, date in enumerate(dates)}
    release_to_index = {label: idx for idx, label in enumerate(RELEASE_LABELS)}
    pv_forecast = np.full((len(dates), 4, 144), np.nan, dtype=float)
    for row in forecast_long.itertuples(index=False):
        issue_date = pd.Timestamp(row.issue_date).normalize()
        if issue_date not in date_to_index or row.issue_time not in release_to_index:
            continue
        minutes_from_day_start = int(
            (pd.Timestamp(row.target_timestamp) - issue_date).total_seconds() // 60
        )
        if 10 <= minutes_from_day_start <= 1440:
            slot = minutes_from_day_start // 10 - 1
            pv_forecast[
                date_to_index[issue_date], release_to_index[row.issue_time], slot
            ] = float(row.pv_forecast_kw) * config.step_hours

    for release_idx, release_slot in enumerate(RELEASE_SLOTS):
        if np.isnan(pv_forecast[:, release_idx, release_slot:]).any():
            missing = int(np.isnan(pv_forecast[:, release_idx, release_slot:]).sum())
            raise ValueError(
                f"{RELEASE_LABELS[release_idx]} 预报在可用时域内缺少 {missing} 个值"
            )

    if not np.isfinite(load).all() or not np.isfinite(pv).all():
        raise ValueError("负载或光伏数据含非有限值")
    if (load < 0).any() or (pv < 0).any() or (variable_price < 0).any():
        raise ValueError("基础求解器要求负载、光伏和电价均非负")

    return DataBundle(
        typical=typical,
        dates=dates,
        load_kwh=load,
        pv_kwh=pv,
        net_kwh=load - pv,
        variable_price=variable_price,
        fixed_price=fixed_price,
        pv_forecast_kwh=pv_forecast,
    )


def solve_question1(data: DataBundle, config: ModelConfig) -> tuple[PlanResult, pd.DataFrame]:
    """Step 3A: solve the deterministic typical-day energy-flow model."""

    net = (
        data.typical["load_energy_kwh"].to_numpy(dtype=float)
        - data.typical["pv_forecast_energy_kwh"].to_numpy(dtype=float)
    )
    plan = solve_energy_plan(
        net,
        data.fixed_price,
        config.storage_initial_kwh,
        config,
        terminal_target_kwh=config.storage_initial_kwh,
    )
    frame = data.typical[
        [
            "interval_index",
            "interval_start",
            "interval_end",
            "load_energy_kwh",
            "pv_forecast_energy_kwh",
        ]
    ].copy()
    frame["purchase_kwh"] = plan.purchase_kwh
    frame["charge_kwh"] = plan.charge_kwh
    frame["discharge_kwh"] = plan.discharge_kwh
    frame["curtail_kwh"] = plan.curtail_kwh
    frame["storage_kwh"] = plan.storage_kwh
    frame["price_yuan_per_kwh"] = data.fixed_price
    return plan, frame


def _empty_mode_arrays(days: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros((days, 144), dtype=float)
        for name in [
            "original",
            "accepted",
            "used",
            "emergency",
            "charge",
            "discharge",
            "curtail",
            "storage",
            "center",
            "risk",
        ]
    }


def solve_day_ahead_mode(
    data: DataBundle,
    config: ModelConfig,
    *,
    price_matrix: np.ndarray,
    name: str,
) -> ModeResult:
    """Step 3B/3D: causal day-ahead plan and real-time recourse."""

    first = int(np.flatnonzero(data.dates >= pd.Timestamp(config.output_start_date))[0])
    output_dates = data.dates[first:]
    days = len(output_dates)
    arrays = _empty_mode_arrays(days)
    storage_start = np.zeros(days)
    daily_rows: list[dict[str, Any]] = []
    energy = config.storage_initial_kwh

    for out_idx, day_idx in enumerate(range(first, len(data.dates))):
        forecast = causal_profile_forecast(data.net_kwh, data.dates, day_idx, config)
        price = np.asarray(price_matrix[day_idx], dtype=float)
        plan = solve_energy_plan(
            forecast.risk_kwh,
            price,
            energy,
            config,
            terminal_target_kwh=energy,
        )
        storage_start[out_idx] = energy
        dispatch = simulate_causal_dispatch(
            data.net_kwh[day_idx], plan.purchase_kwh, energy, config
        )
        energy = float(dispatch.storage_kwh[-1])

        arrays["original"][out_idx] = plan.purchase_kwh
        arrays["accepted"][out_idx] = plan.purchase_kwh
        arrays["used"][out_idx] = dispatch.used_plan_kwh
        arrays["emergency"][out_idx] = dispatch.emergency_kwh
        arrays["charge"][out_idx] = dispatch.charge_kwh
        arrays["discharge"][out_idx] = dispatch.discharge_kwh
        arrays["curtail"][out_idx] = dispatch.curtail_kwh
        arrays["storage"][out_idx] = dispatch.storage_kwh
        arrays["center"][out_idx] = forecast.center_kwh
        arrays["risk"][out_idx] = forecast.risk_kwh

        plan_cost = float(np.sum(price * plan.purchase_kwh))
        emergency_cost = float(
            np.sum(
                config.emergency_price_multiplier * price * dispatch.emergency_kwh
            )
        )
        daily_rows.append(
            {
                "date": output_dates[out_idx],
                "plan_energy_kwh": float(plan.purchase_kwh.sum()),
                "used_plan_energy_kwh": float(dispatch.used_plan_kwh.sum()),
                "emergency_energy_kwh": float(dispatch.emergency_kwh.sum()),
                "charge_energy_kwh": float(dispatch.charge_kwh.sum()),
                "discharge_energy_kwh": float(dispatch.discharge_kwh.sum()),
                "curtail_energy_kwh": float(dispatch.curtail_kwh.sum()),
                "storage_start_kwh": storage_start[out_idx],
                "storage_end_kwh": energy,
                "plan_cost_yuan": plan_cost,
                "adjustment_cost_yuan": 0.0,
                "emergency_cost_yuan": emergency_cost,
                "total_cost_yuan": plan_cost + emergency_cost,
                "forecast_mae_kwh": float(
                    np.mean(np.abs(data.net_kwh[day_idx] - forecast.center_kwh))
                ),
                "max_balance_residual_kwh": dispatch.max_balance_residual_kwh,
                "gate_count": 0,
            }
        )
        if (out_idx + 1) % 50 == 0 or out_idx + 1 == days:
            print(f"{name}: 已完成 {out_idx + 1}/{days} 日", flush=True)

    return ModeResult(
        name=name,
        dates=output_dates,
        original_plan_kwh=arrays["original"],
        accepted_plan_kwh=arrays["accepted"],
        used_plan_kwh=arrays["used"],
        emergency_kwh=arrays["emergency"],
        charge_kwh=arrays["charge"],
        discharge_kwh=arrays["discharge"],
        curtail_kwh=arrays["curtail"],
        storage_kwh=arrays["storage"],
        storage_start_kwh=storage_start,
        forecast_center_kwh=arrays["center"],
        forecast_risk_kwh=arrays["risk"],
        price_yuan_per_kwh=price_matrix[first:].copy(),
        daily=pd.DataFrame(daily_rows),
        gates=pd.DataFrame(),
    )


def solve_rolling_mode(
    data: DataBundle,
    config: ModelConfig,
    *,
    price_matrix: np.ndarray,
    name: str,
    price_case: str,
) -> ModeResult:
    """Step 3C/3E: solve the four-release value-gated rolling mode."""

    first = int(np.flatnonzero(data.dates >= pd.Timestamp(config.output_start_date))[0])
    output_dates = data.dates[first:]
    days = len(output_dates)
    arrays = _empty_mode_arrays(days)
    storage_start = np.zeros(days)
    daily_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    energy = config.storage_initial_kwh

    for out_idx, day_idx in enumerate(range(first, len(data.dates))):
        date = output_dates[out_idx]
        price = np.asarray(price_matrix[day_idx], dtype=float)
        initial_forecast = intraday_net_forecast(
            data.load_kwh,
            data.pv_kwh,
            data.pv_forecast_kwh,
            data.dates,
            day_idx,
            release_index=0,
            release_slot=0,
            config=config,
        )
        plan0 = solve_energy_plan(
            initial_forecast.risk_kwh,
            price,
            energy,
            config,
            terminal_target_kwh=energy,
        )
        current_plan = plan0.purchase_kwh.copy()
        storage_start[out_idx] = energy

        used = np.zeros(144)
        emergency = np.zeros(144)
        charge = np.zeros(144)
        discharge = np.zeros(144)
        curtail = np.zeros(144)
        storage = np.zeros(144)
        gates_accepted = 0

        for release_idx, start in enumerate(RELEASE_SLOTS):
            forecast = intraday_net_forecast(
                data.load_kwh,
                data.pv_kwh,
                data.pv_forecast_kwh,
                data.dates,
                day_idx,
                release_index=release_idx,
                release_slot=start,
                config=config,
            )
            if release_idx > 0:
                candidate = solve_energy_plan(
                    forecast.risk_kwh[start:],
                    price[start:],
                    energy,
                    config,
                    terminal_target_kwh=energy,
                )
                scenarios = forecast.scenarios_kwh[:, start:]
                keep_cost, keep_emergency = scenario_future_costs(
                    current_plan[start:],
                    plan0.purchase_kwh[start:],
                    scenarios,
                    price[start:],
                    energy,
                    config,
                )
                new_cost, new_emergency = scenario_future_costs(
                    candidate.purchase_kwh,
                    plan0.purchase_kwh[start:],
                    scenarios,
                    price[start:],
                    energy,
                    config,
                )
                saving = keep_cost - new_cost
                standard_error = (
                    float(np.std(saving, ddof=1) / np.sqrt(len(saving)))
                    if len(saving) > 1
                    else 0.0
                )
                mean_saving = float(np.mean(saving))
                lcb = mean_saving - config.gate_confidence_z * standard_error
                threshold = config.gate_min_saving_ratio * float(
                    np.sum(price * plan0.purchase_kwh)
                )
                keep_tail_emergency = float(
                    np.quantile(keep_emergency, config.conformal_alpha)
                )
                reliability_trigger = bool(
                    keep_tail_emergency > config.gate_emergency_threshold_kwh
                    and float(np.quantile(new_emergency, config.conformal_alpha))
                    <= keep_tail_emergency
                    * (1.0 - config.gate_required_risk_reduction_ratio)
                )
                accepted = bool(lcb > threshold or reliability_trigger)
                if accepted:
                    current_plan[start:] = candidate.purchase_kwh
                    gates_accepted += 1
                if (
                    reliability_trigger and lcb <= threshold
                ):
                    trigger = "可靠性"
                elif lcb > threshold:
                    trigger = "经济性"
                else:
                    trigger = "未触发"
                gate_rows.append(
                    {
                        "date": date,
                        "price_case": price_case,
                        "release_time": RELEASE_LABELS[release_idx],
                        "mean_saving_yuan": mean_saving,
                        "saving_lcb_yuan": lcb,
                        "minimum_saving_yuan": threshold,
                        "keep_tail_emergency_kwh": keep_tail_emergency,
                        "new_tail_emergency_kwh": float(
                            np.quantile(new_emergency, config.conformal_alpha)
                        ),
                        "accepted": accepted,
                        "trigger": trigger,
                    }
                )

            end = RELEASE_SLOTS[release_idx + 1] if release_idx < 3 else 144
            segment = simulate_causal_dispatch(
                data.net_kwh[day_idx, start:end],
                current_plan[start:end],
                energy,
                config,
            )
            used[start:end] = segment.used_plan_kwh
            emergency[start:end] = segment.emergency_kwh
            charge[start:end] = segment.charge_kwh
            discharge[start:end] = segment.discharge_kwh
            curtail[start:end] = segment.curtail_kwh
            storage[start:end] = segment.storage_kwh
            energy = float(segment.storage_kwh[-1])

        arrays["original"][out_idx] = plan0.purchase_kwh
        arrays["accepted"][out_idx] = current_plan
        arrays["used"][out_idx] = used
        arrays["emergency"][out_idx] = emergency
        arrays["charge"][out_idx] = charge
        arrays["discharge"][out_idx] = discharge
        arrays["curtail"][out_idx] = curtail
        arrays["storage"][out_idx] = storage
        arrays["center"][out_idx] = initial_forecast.center_kwh
        arrays["risk"][out_idx] = initial_forecast.risk_kwh

        plan_cost = float(np.sum(price * plan0.purchase_kwh))
        change_cost = adjustment_cost(current_plan, plan0.purchase_kwh, price, config)
        emergency_cost = float(
            np.sum(config.emergency_price_multiplier * price * emergency)
        )
        residual = used + emergency + discharge - charge - curtail - data.net_kwh[day_idx]
        daily_rows.append(
            {
                "date": date,
                "plan_energy_kwh": float(plan0.purchase_kwh.sum()),
                "accepted_plan_energy_kwh": float(current_plan.sum()),
                "used_plan_energy_kwh": float(used.sum()),
                "emergency_energy_kwh": float(emergency.sum()),
                "charge_energy_kwh": float(charge.sum()),
                "discharge_energy_kwh": float(discharge.sum()),
                "curtail_energy_kwh": float(curtail.sum()),
                "storage_start_kwh": storage_start[out_idx],
                "storage_end_kwh": energy,
                "plan_cost_yuan": plan_cost,
                "adjustment_cost_yuan": change_cost,
                "emergency_cost_yuan": emergency_cost,
                "total_cost_yuan": plan_cost + change_cost + emergency_cost,
                "forecast_mae_kwh": float(
                    np.mean(
                        np.abs(data.net_kwh[day_idx] - initial_forecast.center_kwh)
                    )
                ),
                "max_balance_residual_kwh": float(np.max(np.abs(residual))),
                "gate_count": gates_accepted,
            }
        )
        if (out_idx + 1) % 25 == 0 or out_idx + 1 == days:
            print(f"{name}: 已完成 {out_idx + 1}/{days} 日", flush=True)

    return ModeResult(
        name=name,
        dates=output_dates,
        original_plan_kwh=arrays["original"],
        accepted_plan_kwh=arrays["accepted"],
        used_plan_kwh=arrays["used"],
        emergency_kwh=arrays["emergency"],
        charge_kwh=arrays["charge"],
        discharge_kwh=arrays["discharge"],
        curtail_kwh=arrays["curtail"],
        storage_kwh=arrays["storage"],
        storage_start_kwh=storage_start,
        forecast_center_kwh=arrays["center"],
        forecast_risk_kwh=arrays["risk"],
        price_yuan_per_kwh=price_matrix[first:].copy(),
        daily=pd.DataFrame(daily_rows),
        gates=pd.DataFrame(gate_rows),
    )


def mode_summary(mode: ModeResult) -> dict[str, Any]:
    daily = mode.daily
    plan_cost = float(daily["plan_cost_yuan"].sum())
    adjustment = float(daily["adjustment_cost_yuan"].sum())
    emergency = float(daily["emergency_cost_yuan"].sum())
    return {
        "model": mode.name,
        "days": int(len(daily)),
        "plan_cost_yuan": plan_cost,
        "adjustment_cost_yuan": adjustment,
        "emergency_cost_yuan": emergency,
        "total_cost_yuan": plan_cost + adjustment + emergency,
        "plan_energy_kwh": float(daily["plan_energy_kwh"].sum()),
        "emergency_energy_kwh": float(daily["emergency_energy_kwh"].sum()),
        "charge_energy_kwh": float(daily["charge_energy_kwh"].sum()),
        "discharge_energy_kwh": float(daily["discharge_energy_kwh"].sum()),
        "max_balance_residual_kwh": float(daily["max_balance_residual_kwh"].max()),
        "minimum_storage_kwh": float(mode.storage_kwh.min()),
        "maximum_storage_kwh": float(mode.storage_kwh.max()),
        "accepted_gate_count": (
            int(mode.gates["accepted"].sum()) if len(mode.gates) else 0
        ),
    }


def interval_label(slot: int) -> tuple[str, str]:
    start_minutes = slot * 10
    end_minutes = (slot + 1) * 10
    start = f"{start_minutes // 60:02d}:{start_minutes % 60:02d}"
    end = (
        "24:00"
        if end_minutes == 1440
        else f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
    )
    return start, end


def mode_interval_frame(mode: ModeResult, actual_net_kwh: np.ndarray) -> pd.DataFrame:
    days = len(mode.dates)
    return pd.DataFrame(
        {
            "date": np.repeat(mode.dates.to_numpy(), 144),
            "interval_index": np.tile(np.arange(1, 145), days),
            "interval_start": np.tile([interval_label(t)[0] for t in range(144)], days),
            "interval_end": np.tile([interval_label(t)[1] for t in range(144)], days),
            "price_yuan_per_kwh": mode.price_yuan_per_kwh.reshape(-1),
            "actual_net_load_kwh": actual_net_kwh.reshape(-1),
            "forecast_center_kwh": mode.forecast_center_kwh.reshape(-1),
            "forecast_risk_kwh": mode.forecast_risk_kwh.reshape(-1),
            "original_plan_kwh": mode.original_plan_kwh.reshape(-1),
            "accepted_plan_kwh": mode.accepted_plan_kwh.reshape(-1),
            "used_plan_kwh": mode.used_plan_kwh.reshape(-1),
            "emergency_kwh": mode.emergency_kwh.reshape(-1),
            "charge_kwh": mode.charge_kwh.reshape(-1),
            "discharge_kwh": mode.discharge_kwh.reshape(-1),
            "curtail_kwh": mode.curtail_kwh.reshape(-1),
            "storage_kwh": mode.storage_kwh.reshape(-1),
        }
    )


def _copy_template(root: Path, output_dir: Path, name: str) -> Path:
    destination = output_dir / name
    shutil.copy2(root / "data" / "result-data" / name, destination)
    return destination


def _write_plan_sheet(
    worksheet: openpyxl.worksheet.worksheet.Worksheet,
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    prices: np.ndarray,
    cost_values: np.ndarray | None = None,
) -> None:
    for idx, date in enumerate(dates):
        row = idx + 2
        worksheet.cell(row, 1).value = date.to_pydatetime()
        for t in range(144):
            worksheet.cell(row, t + 2).value = float(values[idx, t])
        worksheet.cell(row, 146).value = float(values[idx].sum())
        worksheet.cell(row, 147).value = (
            float(cost_values[idx])
            if cost_values is not None
            else float(np.sum(prices[idx] * values[idx]))
        )
        for col in range(2, 148):
            worksheet.cell(row, col).number_format = "0.000"


def _parse_date(value: Any) -> pd.Timestamp | None:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, str) and value.strip() and "⁝" not in value:
        try:
            return pd.Timestamp(value).normalize()
        except (ValueError, TypeError):
            return None
    return None


def _write_charge_sheet(
    worksheet: openpyxl.worksheet.worksheet.Worksheet,
    mode: ModeResult,
) -> None:
    date_to_idx = {date.normalize(): idx for idx, date in enumerate(mode.dates)}
    current_date: pd.Timestamp | None = None
    segment_counter = 0
    for row in range(2, worksheet.max_row + 1):
        parsed = _parse_date(worksheet.cell(row, 1).value)
        if parsed is not None:
            current_date = parsed
            segment_counter = 0
        elif worksheet.cell(row, 1).value not in (None, ""):
            current_date = None
        if current_date not in date_to_idx:
            continue
        idx = date_to_idx[current_date]
        label = str(worksheet.cell(row, 2).value or "")
        try:
            start_hour = int(label.split(":")[0])
            segment = start_hour // 4
        except (ValueError, IndexError):
            segment = segment_counter
        if not 0 <= segment < 6:
            continue
        start = segment * 24
        end = start + 24
        worksheet.cell(row, 3).value = float(mode.charge_kwh[idx, start:end].sum())
        worksheet.cell(row, 4).value = float(mode.discharge_kwh[idx, start:end].sum())
        worksheet.cell(row, 3).number_format = "0.000"
        worksheet.cell(row, 4).number_format = "0.000"
        time_value = worksheet.cell(row, 5).value
        if isinstance(time_value, time):
            if time_value.hour == 0 and time_value.minute == 0:
                worksheet.cell(row, 6).value = float(mode.storage_start_kwh[idx])
        elif isinstance(time_value, str):
            if time_value.startswith("0:00") or time_value.startswith("00:00"):
                worksheet.cell(row, 6).value = float(mode.storage_start_kwh[idx])
            elif time_value.startswith("24:00"):
                worksheet.cell(row, 6).value = float(mode.storage_kwh[idx, -1])
        if segment == 1 and worksheet.cell(row, 6).value is None:
            worksheet.cell(row, 6).value = float(mode.storage_kwh[idx, -1])
        worksheet.cell(row, 6).number_format = "0.000"
        segment_counter += 1


def _emergency_events(values: np.ndarray, tolerance: float = 1.0e-6) -> list[tuple[str, float]]:
    events: list[tuple[str, float]] = []
    active = np.flatnonzero(values > tolerance)
    if not len(active):
        return events
    start = int(active[0])
    previous = start
    for slot in active[1:]:
        slot = int(slot)
        if slot != previous + 1:
            events.append(
                (
                    f"{interval_label(start)[0]}-{interval_label(previous)[1]}",
                    float(values[start : previous + 1].sum()),
                )
            )
            start = slot
        previous = slot
    events.append(
        (
            f"{interval_label(start)[0]}-{interval_label(previous)[1]}",
            float(values[start : previous + 1].sum()),
        )
    )
    return events


def _write_emergency_sheet(
    worksheet: openpyxl.worksheet.worksheet.Worksheet,
    mode: ModeResult,
) -> None:
    date_to_idx = {date.normalize(): idx for idx, date in enumerate(mode.dates)}
    groups: list[tuple[pd.Timestamp, list[int]]] = []
    current_date: pd.Timestamp | None = None
    rows: list[int] = []
    for row in range(2, worksheet.max_row + 1):
        parsed = _parse_date(worksheet.cell(row, 1).value)
        if parsed is not None:
            if current_date is not None and rows:
                groups.append((current_date, rows))
            current_date = parsed
            rows = [row]
        elif worksheet.cell(row, 1).value in (None, "") and current_date is not None:
            rows.append(row)
        else:
            if current_date is not None and rows:
                groups.append((current_date, rows))
            current_date = None
            rows = []
    if current_date is not None and rows:
        groups.append((current_date, rows))

    for date, group_rows in groups:
        if date not in date_to_idx:
            continue
        events = _emergency_events(mode.emergency_kwh[date_to_idx[date]])
        if not events:
            events = [("无", 0.0)]
        if len(events) > len(group_rows):
            kept = events[: len(group_rows) - 1]
            rest = events[len(group_rows) - 1 :]
            events = kept + [("其余时段合计", float(sum(value for _, value in rest)))]
        for row, event in zip(group_rows, events, strict=False):
            worksheet.cell(row, 2).value = event[0]
            worksheet.cell(row, 3).value = event[1]
            worksheet.cell(row, 3).number_format = "0.000"


def write_result1_workbook(
    root: Path,
    output_dir: Path,
    plan: PlanResult,
) -> Path:
    path = _copy_template(root, output_dir, "result1.xlsx")
    workbook = openpyxl.load_workbook(path)
    purchase_sheet = workbook.worksheets[0]
    for t in range(144):
        purchase_sheet.cell(t + 2, 2).value = float(plan.purchase_kwh[t])
        purchase_sheet.cell(t + 2, 2).number_format = "0.000"
    charge_sheet = workbook.worksheets[1]
    for segment in range(6):
        row = segment + 2
        start = segment * 24
        end = start + 24
        charge_sheet.cell(row, 2).value = float(plan.charge_kwh[start:end].sum())
        charge_sheet.cell(row, 3).value = float(plan.discharge_kwh[start:end].sum())
        charge_sheet.cell(row, 2).number_format = "0.000"
        charge_sheet.cell(row, 3).number_format = "0.000"
    charge_sheet.cell(2, 5).value = 6000.0
    charge_sheet.cell(3, 5).value = float(plan.storage_kwh[-1])
    workbook.save(path)
    return path


def write_mode_workbook(
    root: Path,
    output_dir: Path,
    template_name: str,
    mode: ModeResult,
    *,
    rolling: bool,
) -> Path:
    path = _copy_template(root, output_dir, template_name)
    workbook = openpyxl.load_workbook(path)
    _write_plan_sheet(
        workbook.worksheets[0],
        mode.dates,
        mode.original_plan_kwh,
        mode.price_yuan_per_kwh,
        mode.daily["plan_cost_yuan"].to_numpy(),
    )
    if rolling:
        adjusted_cost = (
            mode.daily["plan_cost_yuan"] + mode.daily["adjustment_cost_yuan"]
        ).to_numpy()
        _write_plan_sheet(
            workbook.worksheets[1],
            mode.dates,
            mode.accepted_plan_kwh,
            mode.price_yuan_per_kwh,
            adjusted_cost,
        )
        charge_sheet = workbook.worksheets[2]
        emergency_sheet = workbook.worksheets[3]
    else:
        charge_sheet = workbook.worksheets[1]
        emergency_sheet = workbook.worksheets[2]
    _write_charge_sheet(charge_sheet, mode)
    _write_emergency_sheet(emergency_sheet, mode)
    workbook.save(path)
    return path


def write_solution_note(
    output_path: Path,
    config: ModelConfig,
    q1_baseline: float,
    q1_cost: float,
    summary: pd.DataFrame,
    captions: list[str],
) -> None:
    q1_saving = 100.0 * (q1_baseline - q1_cost) / q1_baseline
    by_model = summary.set_index("model")
    q2 = by_model.loc["Q2固定价日前"]
    q3 = by_model.loc["Q3固定价滚动"]
    q3_change = (
        100.0
        * (q2["total_cost_yuan"] - q3["total_cost_yuan"])
        / q2["total_cost_yuan"]
    )
    text = f"""# C题模型求解与结果说明

## 一、运行方式

在项目根目录执行：uv run python -m sxjm.solve_all_models

程序严格按以下顺序运行。

### 1. 数据输入

- 读取 data/processed/typical_day_10min.csv。
- 读取 data/processed/year_actuals_tariff_10min.csv。
- 读取 data/processed/pv_forecasts_10min_linear.csv。
- 读取并复制 data/result-data 中五个Excel模板。

注意：输入功率已经在预处理阶段乘以1/6小时转为kWh。内部时段按
00:00—00:10至23:50—24:00排列；写入赛事模板时保持原模板的144个数据位置，
不改动模板给出的文字标签。

### 2. 参数初始化

- 储能范围：{config.storage_min_kwh:.0f}—{config.storage_max_kwh:.0f} kWh。
- 单段充放电上限：{config.max_interval_energy_kwh:.3f} kWh。
- 充、放电效率：{config.eta_charge:.2f}、{config.eta_discharge:.2f}。
- 风险分位数 alpha：{config.risk_alpha:.2f}。
- 保形覆盖参数：{config.conformal_alpha:.2f}。
- 鲁棒裕度系数：{config.robustness_radius:.2f}。
- 调减退款比例 rho：{config.down_refund_ratio:.1f}。

注意：risk_alpha和conformal_alpha必须在(0,1)内；robustness_radius和
down_refund_ratio必须在[0,1]内。参数只能用决策日前的滚动验证集调试，
不能用全年结果反向选择。

### 3. 模型调用

- 问题1调用稀疏HiGHS线性规划，终端SOC严格等于初始SOC。
- 问题2每天用过去{config.recent_window_days}天构造中心预测、80%净负荷分位和
  保形裕度，再求日前计划；实际执行逐时段使用当前真实净负荷，不读取未来值。
- 问题3使用附件3在0:00、6:00、12:00、18:00发布的光伏预报。每次同时评价
  保持方案和候选方案，只有收益下置信界超过门槛或缺口风险超过阈值才调整。
- 问题4按“当日电价曲线在0:00已知”口径，用附件4逐日波动价格分别重算
  问题2和问题3。

注意：问题2—4采用可运行的分解近似。Wasserstein-DRO/CVaR的作用通过
“经济临界分位数＋保形误差带＋鲁棒裕度＋门控场景压力测试”实现；这不是完整
高维Wasserstein对偶MILP。实际SOC逐日连续传递，规划端的终端SOC中性约束只
防止24小时有限时域无偿透支。

### 4. 结果输出

- 五个填充后的Excel结果模板。
- 每问的逐时段CSV和逐日CSV。
- 门控事件、模型汇总、JSON验证指标。
- 具有标题、坐标轴、图例和图下结论的PNG图。

## 二、主要结果

- 问题1无储能基线费用为 {q1_baseline:,.2f} 元，优化费用为
  {q1_cost:,.2f} 元，下降 {q1_saving:.2f}%。
- 问题2评价期总费用为 {q2['total_cost_yuan']:,.2f} 元，其中紧急购电费为
  {q2['emergency_cost_yuan']:,.2f} 元。
- 问题3评价期总费用为 {q3['total_cost_yuan']:,.2f} 元，相对问题2下降
  {q3_change:.2f}%。

## 三、图表结论

""" + "\n".join(f"- {caption}" for caption in captions) + "\n"
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="求解C题四问并输出结果和图表")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "model_solution",
        help="结果目录，默认 outputs/model_solution",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = (
        (root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
    )
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig()
    config.validate()
    print("参数校验通过", flush=True)

    data = load_inputs(root, config)
    print("数据输入完成", flush=True)

    q1_plan, q1_frame = solve_question1(data, config)
    print("问题1求解完成", flush=True)

    fixed_prices = np.repeat(data.fixed_price[None, :], len(data.dates), axis=0)
    q2 = solve_day_ahead_mode(
        data, config, price_matrix=fixed_prices, name="Q2固定价日前"
    )
    q3 = solve_rolling_mode(
        data,
        config,
        price_matrix=fixed_prices,
        name="Q3固定价滚动",
        price_case="固定电价",
    )
    q42 = solve_day_ahead_mode(
        data, config, price_matrix=data.variable_price, name="Q4-2波动价日前"
    )
    q43 = solve_rolling_mode(
        data,
        config,
        price_matrix=data.variable_price,
        name="Q4-3波动价滚动",
        price_case="波动电价",
    )
    modes = [q2, q3, q42, q43]

    q1_frame.to_csv(
        output_dir / "question1_dispatch.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    first = int(np.flatnonzero(data.dates >= pd.Timestamp(config.output_start_date))[0])
    for mode, stem in zip(
        modes,
        ["question2", "question3", "question4_2", "question4_3"],
        strict=True,
    ):
        mode.daily.to_csv(
            output_dir / f"{stem}_daily.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        mode_interval_frame(mode, data.net_kwh[first:]).to_csv(
            output_dir / f"{stem}_interval.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
    gates = pd.concat([q3.gates, q43.gates], ignore_index=True)
    gates.to_csv(
        output_dir / "gate_events.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    summary = pd.DataFrame([mode_summary(mode) for mode in modes])
    summary.to_csv(
        output_dir / "model_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    q1_baseline = float(
        np.sum(
            data.fixed_price
            * np.maximum(
                data.typical["load_energy_kwh"].to_numpy(dtype=float)
                - data.typical["pv_forecast_energy_kwh"].to_numpy(dtype=float),
                0.0,
            )
        )
    )
    q1_cost = float(np.sum(data.fixed_price * q1_plan.purchase_kwh))
    validation = {
        "config": asdict(config),
        "question1": {
            "solver_status": q1_plan.solver_status,
            "baseline_cost_yuan": q1_baseline,
            "optimized_cost_yuan": q1_cost,
            "max_balance_residual_kwh": q1_plan.max_balance_residual_kwh,
            "simultaneous_flow_kwh": q1_plan.simultaneous_flow_kwh,
            "storage_start_kwh": config.storage_initial_kwh,
            "storage_end_kwh": float(q1_plan.storage_kwh[-1]),
        },
        "models": summary.to_dict(orient="records"),
        "template_mapping_note": (
            "内部按00:00-00:10至23:50-24:00排列；按144个数据位置写入模板，"
            "原模板文字标签保留不变。"
        ),
        "implementation_scope": (
            "Q2-Q4使用经济临界分位数、在线误差带和场景门控实现DRO/CVaR的"
            "可运行分解近似，不声称为完整高维Wasserstein对偶MILP。"
        ),
    }
    (output_dir / "solution_summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_result1_workbook(root, output_dir, q1_plan)
    write_mode_workbook(root, output_dir, "result2.xlsx", q2, rolling=False)
    write_mode_workbook(root, output_dir, "result3.xlsx", q3, rolling=True)
    write_mode_workbook(root, output_dir, "result4-2.xlsx", q42, rolling=False)
    write_mode_workbook(root, output_dir, "result4-3.xlsx", q43, rolling=True)
    print("Excel结果模板已填充", flush=True)

    captions: list[str] = []
    captions.append(
        plot_question1(
            data.typical,
            q1_frame,
            figures_dir / "figure1_question1_dispatch.png",
            q1_baseline,
            q1_cost,
        )
    )
    representative = pd.Timestamp(config.representative_date)
    rep_idx = int(np.flatnonzero(q2.dates == representative)[0])
    data_idx = int(np.flatnonzero(data.dates == representative)[0])
    captions.append(
        plot_question2_forecast(
            representative,
            data.net_kwh[data_idx],
            q2.forecast_center_kwh[rep_idx],
            q2.forecast_risk_kwh[rep_idx],
            q2.original_plan_kwh[rep_idx],
            q2.emergency_kwh[rep_idx],
            figures_dir / "figure2_question2_forecast.png",
        )
    )
    captions.append(
        plot_monthly_costs(
            q2.daily,
            figures_dir / "figure3_question2_monthly_cost.png",
            3,
            "问题2：月度购电费用构成",
        )
    )
    captions.append(
        plot_monthly_costs(
            q3.daily,
            figures_dir / "figure4_question3_monthly_cost.png",
            4,
            "问题3：滚动调整后的月度费用构成",
        )
    )
    captions.append(
        plot_strategy_comparison(
            summary,
            gates,
            figures_dir / "figure5_strategy_comparison.png",
        )
    )
    write_solution_note(
        output_dir / "模型求解与结果说明.md",
        config,
        q1_baseline,
        q1_cost,
        summary,
        captions,
    )
    print(summary.to_string(index=False), flush=True)
    print(f"全部结果已写入：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
