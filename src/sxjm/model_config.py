"""Configuration and validation for the C-problem solvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """All numerical assumptions used by the executable model.

    In the verified solver risk_alpha is the CVaR confidence level.
    The legacy heuristic also uses it as a one-sided planning quantile.
    Defaults are explicit experimental choices, not independently tuned optima.
    """

    step_hours: float = 1.0 / 6.0
    storage_min_kwh: float = 1200.0
    storage_max_kwh: float = 10800.0
    storage_initial_kwh: float = 6000.0
    storage_power_kw: float = 5000.0
    eta_charge: float = 0.90
    eta_discharge: float = 0.90
    emergency_price_multiplier: float = 5.0
    adjustment_up_multiplier: float = 1.5
    adjustment_down_penalty_multiplier: float = 0.5
    down_refund_ratio: float = 0.0
    risk_alpha: float = 0.80
    conformal_alpha: float = 0.90
    robustness_radius: float = 0.15
    recent_window_days: int = 56
    same_weekday_count: int = 8
    throughput_penalty_yuan_per_kwh: float = 1.0e-6
    gate_confidence_z: float = 1.645
    gate_min_saving_ratio: float = 0.001
    gate_emergency_threshold_kwh: float = 50.0
    gate_required_risk_reduction_ratio: float = 0.05
    numerical_tolerance: float = 1.0e-6
    main_price_information_case: str = "known_day_ahead"
    output_start_date: str = "2025-02-01"
    representative_date: str = "2025-03-20"

    @property
    def max_interval_energy_kwh(self) -> float:
        return self.storage_power_kw * self.step_hours

    def validate(self) -> None:
        """Raise a clear error before any optimization is started."""

        probability_parameters = {
            "risk_alpha": self.risk_alpha,
            "conformal_alpha": self.conformal_alpha,
        }
        for name, value in probability_parameters.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} 必须在 (0, 1) 内，当前值为 {value}")
        if not 0.0 <= self.robustness_radius <= 1.0:
            raise ValueError("robustness_radius 必须在 [0, 1] 内调试")
        if not 0.0 <= self.down_refund_ratio <= 1.0:
            raise ValueError("down_refund_ratio 必须在 [0, 1] 内")
        if not 0.0 < self.eta_charge <= 1.0:
            raise ValueError("eta_charge 必须在 (0, 1] 内")
        if not 0.0 < self.eta_discharge <= 1.0:
            raise ValueError("eta_discharge 必须在 (0, 1] 内")
        if self.storage_min_kwh >= self.storage_max_kwh:
            raise ValueError("储能下限必须小于上限")
        if not self.storage_min_kwh <= self.storage_initial_kwh <= self.storage_max_kwh:
            raise ValueError("初始储电量必须位于运行区间内")
        if self.storage_power_kw <= 0.0 or self.step_hours <= 0.0:
            raise ValueError("储能功率和时段长度必须为正")
        if self.recent_window_days < 7:
            raise ValueError("recent_window_days 至少为 7，才能刻画周周期")
        if self.same_weekday_count < 1:
            raise ValueError("same_weekday_count 必须为正整数")
        if self.emergency_price_multiplier <= 1.0:
            raise ValueError("紧急购电倍率应大于普通购电倍率 1")
        if self.gate_min_saving_ratio < 0.0:
            raise ValueError("gate_min_saving_ratio 不能为负")
        if self.gate_emergency_threshold_kwh < 0.0:
            raise ValueError("gate_emergency_threshold_kwh 不能为负")
        if not 0.0 <= self.gate_required_risk_reduction_ratio <= 1.0:
            raise ValueError("gate_required_risk_reduction_ratio 必须在 [0, 1] 内")
