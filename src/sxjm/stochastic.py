"""Finite-support Wasserstein mean-CVaR program, with causal battery controls.

This is an explicitly restricted policy approximation to the multistage model:
q, charge, discharge and SOC are common to all scenarios until the next solve.
The transport ambiguity set is on the supplied support, not all of R^(144).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .model_config import ModelConfig
from .optimization import DispatchResult


@dataclass
class StochasticPlan:
    q: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    storage: np.ndarray
    objective: float
    scenario_cost: np.ndarray
    scenario_emergency: np.ndarray
    residual: float
    used_milp: bool
    mip_gap: float


def transport_distance(net: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Dimensionless daily L1 distance; normalize using only supplied scenarios."""
    net_scale = max(float(np.std(net, axis=0).mean()), 1.0)
    price_scale = max(float(np.std(prices, axis=0).mean()), 0.01)
    distance = np.abs(net[:, None, :] - net[None, :, :]).mean(axis=2) / net_scale
    distance += np.abs(prices[:, None, :] - prices[None, :, :]).mean(axis=2) / price_scale
    return distance


def solve_stochastic_plan(
    net: np.ndarray, prices: np.ndarray, probability: np.ndarray,
    initial: float, config: ModelConfig, *, risk_weight: float = 0.20,
    original: np.ndarray | None = None, fixed_q: np.ndarray | None = None,
) -> StochasticPlan:
    """Min sup_P{E_P[C] + risk_weight*CVaR_alpha,P(C)} on a finite support.

    Transport dual: beta_j >= C_s + lambda*xi_s/(1-alpha) - theta*D_js;
    xi_s >= C_s-zeta; objective = lambda*zeta+epsilon*theta+sum pi_j beta_j.
    Final SOC is planned to 6000 kWh, never reset in the actual simulation.
    Up/down cost is inside the optimization, always relative to original q0.
    LP relaxation is accepted only if charge/discharge complementarity holds;
    otherwise binary modes are activated and the same problem is solved as MILP.
    """
    config.validate()
    net, prices, probability = map(lambda x: np.asarray(x, dtype=float), (net, prices, probability))
    if net.ndim != 2 or prices.shape != net.shape:
        raise ValueError("Scenarios and prices must have matching S x T shapes")
    scenarios, horizon = net.shape
    if probability.shape != (scenarios,) or np.any(probability <= 0) or not np.isclose(probability.sum(), 1):
        raise ValueError("Scenario probabilities must be positive and sum to one")
    if not np.isfinite(net).all() or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Finite scenarios and strictly positive prices required")
    if not 0 <= risk_weight <= 1:
        raise ValueError("risk_weight must be in [0,1]")
    if not config.storage_min_kwh - 1e-6 <= initial <= config.storage_max_kwh + 1e-6:
        raise ValueError("Initial SOC outside physical limits")
    initial = float(np.clip(initial, config.storage_min_kwh, config.storage_max_kwh))
    q = np.arange(horizon)
    c, d, e = q + horizon, q + 2*horizon, q + 3*horizon
    h = np.arange(scenarios*horizon).reshape(scenarios, horizon) + 4*horizon
    cursor = 4*horizon + scenarios*horizon
    up, down = np.arange(horizon)+cursor, np.arange(horizon)+cursor+horizon
    cursor += 2*horizon
    loss, xi, beta = [np.arange(scenarios)+cursor+k*scenarios for k in range(3)]
    zeta, theta = cursor+3*scenarios, cursor+3*scenarios+1
    binary = np.arange(horizon)+theta+1
    count = int(binary[-1]+1)
    objective = np.zeros(count)
    objective[beta] = probability
    objective[zeta], objective[theta] = risk_weight, config.robustness_radius
    objective[c] = objective[d] = config.throughput_penalty_yuan_per_kwh
    bounds = [(0., None) for _ in range(count)]
    max_flow = config.max_interval_energy_kwh
    for t in range(horizon):
        bounds[c[t]] = bounds[d[t]] = (0., max_flow)
        bounds[e[t]] = (config.storage_min_kwh, config.storage_max_kwh)
        bounds[binary[t]] = (0., 1.)
        if fixed_q is not None:
            value = float(max(0., fixed_q[t]))
            bounds[q[t]] = (value, value)
        if original is None:
            bounds[up[t]] = bounds[down[t]] = (0., 0.)
        else:
            bounds[down[t]] = (0., float(max(0., original[t])))
    for index in np.r_[loss, beta, zeta]:
        bounds[index] = (None, None)
    eq_rows, eq_cols, eq_values, eq_rhs = [], [], [], []
    ub_rows, ub_cols, ub_values, ub_rhs = [], [], [], []

    def add(indices, values, rhs, equal=False):
        rows, cols, vals, right = (eq_rows, eq_cols, eq_values, eq_rhs) if equal else (ub_rows, ub_cols, ub_values, ub_rhs)
        row = len(right)
        indices, values = np.asarray(indices).ravel(), np.asarray(values).ravel()
        rows.extend([row]*len(indices)); cols.extend(indices.tolist()); vals.extend(values.tolist())
        right.append(float(rhs))

    for t in range(horizon):
        indices, values = [e[t], c[t], d[t]], [1., -config.eta_charge, 1/config.eta_discharge]
        if t:
            indices.append(e[t-1]); values.append(-1.)
        add(indices, values, initial if t == 0 else 0., True)
        # Conservative no battery-discharge disposal under every support path.
        add([d[t], c[t]], [1., -1.], max(0., float(net[:, t].min())))
        add([c[t], binary[t]], [1., -max_flow], 0.)
        add([d[t], binary[t]], [1., max_flow], max_flow)
        if original is not None:
            add([q[t], up[t], down[t]], [1., -1., 1.], original[t], True)
    add([e[-1]], [1.], config.storage_initial_kwh, True)
    for s in range(scenarios):
        for t in range(horizon):
            add([c[t], d[t], q[t], h[s,t]], [1., -1., -1., -1.], -net[s,t])
        indices = list(h[s]) + [loss[s]]
        values = list(config.emergency_price_multiplier*prices[s]) + [-1.]
        constant = 0.
        if original is None:
            indices += list(q); values += list(prices[s])
        else:
            indices += list(up)+list(down)
            values += list(config.adjustment_up_multiplier*prices[s])
            values += list((config.adjustment_down_penalty_multiplier-config.down_refund_ratio)*prices[s])
            constant = float(prices[s]@original)
        add(indices, values, -constant, True)
        add([loss[s], zeta, xi[s]], [1., -1., -1.], 0.)
    distance = transport_distance(net, prices)
    for j in range(scenarios):
        for s in range(scenarios):
            add([loss[s], xi[s], theta, beta[j]], [1., risk_weight/(1-config.risk_alpha), -distance[j,s], -1.], 0.)
    a_eq = coo_matrix((eq_values,(eq_rows,eq_cols)),shape=(len(eq_rhs),count)).tocsr()
    a_ub = coo_matrix((ub_values,(ub_rows,ub_cols)),shape=(len(ub_rhs),count)).tocsr()
    kwargs = dict(A_ub=a_ub, b_ub=ub_rhs, A_eq=a_eq, b_eq=eq_rhs, bounds=bounds, method="highs")
    result = linprog(objective, **kwargs)
    used_milp = False
    if result.success and np.max(np.minimum(result.x[c],result.x[d])) > 1e-6:
        used_milp = True
        integrality = np.zeros(count, dtype=int); integrality[binary] = 1
        result = linprog(objective, integrality=integrality, options={"mip_rel_gap": 1e-6}, **kwargs)
    if not result.success:
        raise RuntimeError(f"Finite-support DRO solve failed: {result.message}")
    values = result.x
    residual = max(float(np.max(np.abs(a_eq@values-eq_rhs))), float(np.max(a_ub@values-ub_rhs)))
    if residual > 1e-4 or np.max(np.minimum(values[c],values[d])) > 1e-5:
        raise RuntimeError(f"Solution failed feasibility check: {residual}")
    # Tight recourse costs are recomputed, since non-worst scenarios can have slack h.
    emergency = np.maximum(net+values[c]-values[d]-values[q],0.)
    if original is None:
        base = prices@values[q]
    else:
        delta = values[q]-original
        base = prices@(original+config.adjustment_up_multiplier*np.maximum(delta,0.)
                       +(config.adjustment_down_penalty_multiplier-config.down_refund_ratio)*np.maximum(-delta,0.))
    scenario_cost = base + config.emergency_price_multiplier*np.sum(prices*emergency,axis=1)
    return StochasticPlan(values[q],values[c],values[d],values[e],float(result.fun),scenario_cost,
                          emergency.sum(axis=1),residual,used_milp,float(getattr(result,"mip_gap",0.) or 0.))


def execute_controls(net, q, charge, discharge, initial, config) -> DispatchResult:
    """Causal projection of prescribed controls onto CURRENT physical conditions.

    No future realized net loads enter any control. Paid but unused q is not sold.
    Emergency energy may fund a scheduled charge; that cost is counted in full.
    """
    net, q = np.asarray(net), np.asarray(q)
    used, emergency, c, d, spill, energy = [np.zeros(len(net)) for _ in range(6)]
    state = float(initial)
    for t, demand in enumerate(net):
        c[t] = min(max(0.,charge[t]),config.max_interval_energy_kwh,max(0.,(config.storage_max_kwh-state)/config.eta_charge))
        d[t] = min(max(0.,discharge[t]),config.max_interval_energy_kwh,max(0.,(state-config.storage_min_kwh)*config.eta_discharge),max(0.,demand))
        required = demand+c[t]-d[t]
        used[t] = min(max(0.,q[t]),max(0.,required))
        emergency[t] = max(0.,required-used[t])
        spill[t] = max(0.,-required)
        state += config.eta_charge*c[t]-d[t]/config.eta_discharge
        energy[t] = state
    residual = used+emergency+d-c-spill-net
    return DispatchResult(used,emergency,c,d,spill,energy,float(np.max(np.abs(residual))))
