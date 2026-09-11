"""Sparse linear optimization and causal dispatch for the C problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .model_config import ModelConfig


@dataclass
class PlanResult:
    purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    storage_kwh: np.ndarray
    objective_yuan: float
    max_balance_residual_kwh: float
    simultaneous_flow_kwh: float
    solver_status: str


@dataclass
class DispatchResult:
    used_plan_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    storage_kwh: np.ndarray
    max_balance_residual_kwh: float


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 含 NaN 或无穷值")
    return result


def solve_energy_plan(
    net_load_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage_start_kwh: float,
    config: ModelConfig,
    *,
    terminal_target_kwh: float | None,
) -> PlanResult:
    """Solve the deterministic risk-adjusted planning LP.

    The terminal target is a rolling-horizon neutrality condition. For
    questions 2--4 it is set to the observed storage at the optimization
    epoch. Actual execution may end elsewhere, and that actual state is
    carried to the next day.
    """

    net = _as_vector(net_load_kwh, "net_load_kwh")
    price = _as_vector(price_yuan_per_kwh, "price_yuan_per_kwh")
    if len(net) != len(price):
        raise ValueError("净负荷和电价长度必须一致")
    if (price < 0.0).any():
        raise ValueError("基础模型不支持负电价；如需支持必须增加计划电弃电变量")
    if not config.storage_min_kwh <= storage_start_kwh <= config.storage_max_kwh:
        raise ValueError("优化初始 SOC 越界")

    horizon = len(net)
    q0 = 0
    c0 = horizon
    d0 = 2 * horizon
    s0 = 3 * horizon
    e0 = 4 * horizon
    variable_count = 5 * horizon

    objective = np.zeros(variable_count, dtype=float)
    objective[q0:c0] = price
    objective[c0:d0] = config.throughput_penalty_yuan_per_kwh
    objective[d0:s0] = config.throughput_penalty_yuan_per_kwh

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []

    # q + d - c - s = net load.
    for t in range(horizon):
        row = len(rhs)
        rows.extend([row, row, row, row])
        cols.extend([q0 + t, c0 + t, d0 + t, s0 + t])
        data.extend([1.0, -1.0, 1.0, -1.0])
        rhs.append(float(net[t]))

    # E_t - E_(t-1) - eta_c*c_t + d_t/eta_d = 0.
    for t in range(horizon):
        row = len(rhs)
        rows.extend([row, row, row])
        cols.extend([c0 + t, d0 + t, e0 + t])
        data.extend([-config.eta_charge, 1.0 / config.eta_discharge, 1.0])
        if t == 0:
            rhs.append(float(storage_start_kwh))
        else:
            rows.append(row)
            cols.append(e0 + t - 1)
            data.append(-1.0)
            rhs.append(0.0)

    if terminal_target_kwh is not None:
        if not config.storage_min_kwh <= terminal_target_kwh <= config.storage_max_kwh:
            raise ValueError("terminal_target_kwh 越界")
        row = len(rhs)
        rows.append(row)
        cols.append(e0 + horizon - 1)
        data.append(1.0)
        rhs.append(float(terminal_target_kwh))

    a_eq = coo_matrix(
        (data, (rows, cols)), shape=(len(rhs), variable_count), dtype=float
    ).tocsr()
    b_eq = np.asarray(rhs, dtype=float)

    max_flow = config.max_interval_energy_kwh
    pv_surplus = np.maximum(-net, 0.0)
    bounds = (
        [(0.0, None)] * horizon
        + [(0.0, max_flow)] * horizon
        + [(0.0, max_flow)] * horizon
        + [(0.0, float(value)) for value in pv_surplus]
        + [(config.storage_min_kwh, config.storage_max_kwh)] * horizon
    )

    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1.0e-7,
            "primal_feasibility_tolerance": 1.0e-7,
        },
    )
    if not result.success:
        raise RuntimeError(f"HiGHS 求解失败：{result.message}")

    values = result.x
    purchase = values[q0:c0]
    charge = values[c0:d0]
    discharge = values[d0:s0]
    curtail = values[s0:e0]
    storage = values[e0:]
    balance = purchase + discharge - charge - curtail - net
    return PlanResult(
        purchase_kwh=purchase,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        storage_kwh=storage,
        objective_yuan=float(result.fun),
        max_balance_residual_kwh=float(np.max(np.abs(balance))),
        simultaneous_flow_kwh=float(np.minimum(charge, discharge).sum()),
        solver_status=str(result.message),
    )


def simulate_causal_dispatch(
    net_load_kwh: np.ndarray,
    planned_purchase_kwh: np.ndarray,
    storage_start_kwh: float,
    config: ModelConfig,
) -> DispatchResult:
    """Execute a fixed purchase plan without looking at future actual values."""

    net = _as_vector(net_load_kwh, "net_load_kwh")
    planned = _as_vector(planned_purchase_kwh, "planned_purchase_kwh")
    if len(net) != len(planned):
        raise ValueError("实际净负荷和计划购电量长度必须一致")
    if (planned < -config.numerical_tolerance).any():
        raise ValueError("计划购电量不能为负")

    horizon = len(net)
    used = np.zeros(horizon)
    emergency = np.zeros(horizon)
    charge = np.zeros(horizon)
    discharge = np.zeros(horizon)
    curtail = np.zeros(horizon)
    storage = np.zeros(horizon)
    residual = np.zeros(horizon)
    energy = float(storage_start_kwh)
    max_flow = config.max_interval_energy_kwh

    for t in range(horizon):
        available_surplus = planned[t] - net[t]
        if available_surplus >= 0.0:
            charge_limit = min(
                max_flow,
                max(0.0, (config.storage_max_kwh - energy) / config.eta_charge),
            )
            charge[t] = min(available_surplus, charge_limit)
            # Use local PV before extracting an already-paid purchase.
            used[t] = min(planned[t], max(0.0, net[t] + charge[t]))
            curtail[t] = max(0.0, used[t] - net[t] - charge[t])
        else:
            used[t] = planned[t]
            deficit = net[t] - used[t]
            discharge_limit = min(
                max_flow,
                max(0.0, (energy - config.storage_min_kwh) * config.eta_discharge),
            )
            discharge[t] = min(deficit, discharge_limit)
            emergency[t] = max(0.0, deficit - discharge[t])

        energy += config.eta_charge * charge[t]
        energy -= discharge[t] / config.eta_discharge
        if energy < config.storage_min_kwh - 1.0e-5:
            raise RuntimeError("因果执行导致 SOC 低于下限")
        if energy > config.storage_max_kwh + 1.0e-5:
            raise RuntimeError("因果执行导致 SOC 高于上限")
        energy = float(np.clip(energy, config.storage_min_kwh, config.storage_max_kwh))
        storage[t] = energy
        residual[t] = (
            used[t]
            + emergency[t]
            + discharge[t]
            - charge[t]
            - curtail[t]
            - net[t]
        )

    return DispatchResult(
        used_plan_kwh=used,
        emergency_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        storage_kwh=storage,
        max_balance_residual_kwh=float(np.max(np.abs(residual))),
    )


def adjustment_cost(
    adjusted_plan_kwh: np.ndarray,
    original_plan_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    config: ModelConfig,
) -> float:
    adjusted = _as_vector(adjusted_plan_kwh, "adjusted_plan_kwh")
    original = _as_vector(original_plan_kwh, "original_plan_kwh")
    price = _as_vector(price_yuan_per_kwh, "price_yuan_per_kwh")
    increase = np.maximum(adjusted - original, 0.0)
    decrease = np.maximum(original - adjusted, 0.0)
    return float(
        np.sum(
            config.adjustment_up_multiplier * price * increase
            + (
                config.adjustment_down_penalty_multiplier
                - config.down_refund_ratio
            )
            * price
            * decrease
        )
    )


def scenario_future_costs(
    planned_purchase_kwh: np.ndarray,
    original_plan_kwh: np.ndarray,
    net_load_scenarios_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage_start_kwh: float,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate adjustment plus emergency costs for gate decisions."""

    scenarios = np.asarray(net_load_scenarios_kwh, dtype=float)
    if scenarios.ndim != 2:
        raise ValueError("net_load_scenarios_kwh 必须为 场景×时段 的二维数组")
    deterministic_adjustment = adjustment_cost(
        planned_purchase_kwh, original_plan_kwh, price_yuan_per_kwh, config
    )
    costs = np.zeros(scenarios.shape[0])
    emergency_energy = np.zeros(scenarios.shape[0])
    price = _as_vector(price_yuan_per_kwh, "price_yuan_per_kwh")
    for omega, net in enumerate(scenarios):
        dispatch = simulate_causal_dispatch(
            net, planned_purchase_kwh, storage_start_kwh, config
        )
        emergency_energy[omega] = float(dispatch.emergency_kwh.sum())
        costs[omega] = deterministic_adjustment + float(
            np.sum(
                config.emergency_price_multiplier
                * price
                * dispatch.emergency_kwh
            )
        )
    return costs, emergency_energy
