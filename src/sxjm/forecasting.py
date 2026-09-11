"""Causal profile, conformal-margin, and intra-day forecast utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model_config import ModelConfig


SCENARIO_MULTIPLIERS = np.asarray(
    [-1.20, -0.85, -0.55, -0.30, 0.0, 0.30, 0.55, 0.85, 1.20],
    dtype=float,
)


@dataclass
class ForecastBundle:
    center_kwh: np.ndarray
    risk_kwh: np.ndarray
    scale_kwh: np.ndarray
    scenarios_kwh: np.ndarray


def _history_indices(
    dates: pd.DatetimeIndex, current_index: int, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, current_index - config.recent_window_days)
    recent = np.arange(start, current_index, dtype=int)
    weekday = dates[current_index].weekday()
    same = np.asarray(
        [idx for idx in recent if dates[idx].weekday() == weekday], dtype=int
    )
    if len(same) > config.same_weekday_count:
        same = same[-config.same_weekday_count :]
    return recent, same


def causal_profile_forecast(
    values_kwh: np.ndarray,
    dates: pd.DatetimeIndex,
    current_index: int,
    config: ModelConfig,
) -> ForecastBundle:
    """Forecast one 144-slot daily profile using only prior days."""

    values = np.asarray(values_kwh, dtype=float)
    if values.ndim != 2:
        raise ValueError("values_kwh 必须为 日期×时段 的二维矩阵")
    recent_indices, same_indices = _history_indices(dates, current_index, config)
    if len(recent_indices) < 7:
        raise ValueError("至少需要 7 个历史日才能生成因果日内预测")

    recent = values[recent_indices]
    recent_center = np.median(recent, axis=0)
    if len(same_indices) >= 2:
        weekday_center = np.median(values[same_indices], axis=0)
        center = 0.65 * weekday_center + 0.35 * recent_center
    else:
        center = recent_center

    residuals = recent - center
    one_sided = np.quantile(residuals, config.risk_alpha, axis=0)
    conformal_scale = np.quantile(
        np.abs(residuals), config.conformal_alpha, axis=0
    )
    dispersion = np.std(residuals, axis=0, ddof=1)
    risk = (
        center
        + np.maximum(one_sided, 0.0)
        + config.robustness_radius * dispersion
    )
    scenarios = center[None, :] + SCENARIO_MULTIPLIERS[:, None] * conformal_scale
    return ForecastBundle(
        center_kwh=center,
        risk_kwh=risk,
        scale_kwh=conformal_scale,
        scenarios_kwh=scenarios,
    )


def intraday_net_forecast(
    load_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    pv_forecast_kwh: np.ndarray,
    dates: pd.DatetimeIndex,
    current_index: int,
    release_index: int,
    release_slot: int,
    config: ModelConfig,
) -> ForecastBundle:
    """Build the net-load forecast available at one release time.

    pv_forecast_kwh has dimensions date × release × time. Only historical
    days are used to estimate forecast errors. Same-day observed load is used
    only before release_slot to apply a bounded level correction.
    """

    load_bundle = causal_profile_forecast(load_kwh, dates, current_index, config)
    load_center = load_bundle.center_kwh.copy()
    if release_slot > 0:
        observed = load_kwh[current_index, :release_slot]
        expected = load_center[:release_slot]
        valid = expected > 1.0e-6
        if valid.any():
            ratio = float(np.median(observed[valid] / expected[valid]))
            load_center[release_slot:] *= np.clip(ratio, 0.80, 1.20)

    pv_prediction = np.asarray(
        pv_forecast_kwh[current_index, release_index], dtype=float
    )
    if np.isnan(pv_prediction[release_slot:]).any():
        raise ValueError(
            f"{dates[current_index].date()} 发布索引 {release_index} 的光伏预报不完整"
        )

    recent_indices, _ = _history_indices(dates, current_index, config)
    start = release_slot
    recent_load = load_kwh[recent_indices, start:]
    load_reference = np.median(recent_load, axis=0)
    load_error = recent_load - load_reference
    past_pv_forecast = pv_forecast_kwh[recent_indices, release_index, start:]
    past_pv_actual = pv_actual_kwh[recent_indices, start:]
    pv_error = past_pv_actual - past_pv_forecast
    net_error = load_error - pv_error

    center = load_center - pv_prediction
    one_sided = np.nanquantile(net_error, config.risk_alpha, axis=0)
    scale = np.nanquantile(
        np.abs(net_error - np.nanmedian(net_error, axis=0)),
        config.conformal_alpha,
        axis=0,
    )
    dispersion = np.nanstd(net_error, axis=0, ddof=1)

    risk = center.copy()
    risk[start:] = (
        center[start:]
        + np.maximum(one_sided, 0.0)
        + config.robustness_radius * dispersion
    )
    full_scale = np.zeros_like(center)
    full_scale[start:] = scale
    scenarios = center[None, :] + SCENARIO_MULTIPLIERS[:, None] * full_scale
    return ForecastBundle(
        center_kwh=center,
        risk_kwh=risk,
        scale_kwh=full_scale,
        scenarios_kwh=scenarios,
    )
