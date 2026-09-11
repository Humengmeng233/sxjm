"""Static figures with titles, axes, legends, and data-derived conclusions."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#2F6B9A",
    "orange": "#E07A3F",
    "green": "#4C956C",
    "red": "#C44E52",
    "purple": "#7D5BA6",
    "gray": "#6C757D",
    "light_blue": "#9EC5E5",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 180,
        }
    )


def _save_with_caption(fig: plt.Figure, path: Path, caption: str) -> None:
    caption_text = textwrap.fill(caption, width=78)
    fig.text(0.5, 0.015, caption_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.98))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_question1(
    typical: pd.DataFrame,
    plan: pd.DataFrame,
    output_path: Path,
    baseline_cost_yuan: float,
    optimized_cost_yuan: float,
) -> str:
    configure_matplotlib()
    hour = np.arange(144) / 6.0
    saving_pct = (
        100.0 * (baseline_cost_yuan - optimized_cost_yuan) / baseline_cost_yuan
        if baseline_cost_yuan > 0
        else 0.0
    )
    caption = (
        f"图 1：储能在低价或光伏富余时段充电、在高价时段放电。"
        f"相较无储能基线，单日购电成本由 {baseline_cost_yuan:,.2f} 元降至"
        f" {optimized_cost_yuan:,.2f} 元，下降 {saving_pct:.2f}%。"
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(
        hour,
        typical["load_energy_kwh"],
        label="负载电量",
        color=COLORS["gray"],
        linewidth=1.4,
    )
    axes[0].plot(
        hour,
        typical["pv_forecast_energy_kwh"],
        label="光伏电量",
        color=COLORS["green"],
        linewidth=1.4,
    )
    axes[0].plot(
        hour,
        plan["purchase_kwh"],
        label="计划购电量",
        color=COLORS["blue"],
        linewidth=1.5,
    )
    axes[0].bar(
        hour,
        plan["charge_kwh"],
        width=0.12,
        label="充电量",
        color=COLORS["light_blue"],
        alpha=0.75,
    )
    axes[0].bar(
        hour,
        -plan["discharge_kwh"],
        width=0.12,
        label="放电量（负向显示）",
        color=COLORS["orange"],
        alpha=0.75,
    )
    axes[0].set_title("问题1：典型日最优购电与储能调度")
    axes[0].set_ylabel("10分钟电量（kWh）")
    axes[0].legend(ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        hour,
        plan["storage_kwh"],
        label="储电量",
        color=COLORS["purple"],
        linewidth=1.7,
    )
    axes[1].axhline(1200, color=COLORS["red"], linestyle="--", label="SOC下限")
    axes[1].axhline(10800, color=COLORS["green"], linestyle="--", label="SOC上限")
    axes[1].set_xlabel("时刻（小时）")
    axes[1].set_ylabel("储电量（kWh）")
    axes[1].set_xlim(0, 24)
    axes[1].legend(ncol=3, loc="upper center")
    axes[1].grid(alpha=0.25)
    _save_with_caption(fig, output_path, caption)
    return caption


def plot_question2_forecast(
    date: pd.Timestamp,
    actual_net_kwh: np.ndarray,
    center_kwh: np.ndarray,
    risk_kwh: np.ndarray,
    plan_kwh: np.ndarray,
    emergency_kwh: np.ndarray,
    output_path: Path,
) -> str:
    configure_matplotlib()
    hour = np.arange(144) / 6.0
    mae = float(np.mean(np.abs(actual_net_kwh - center_kwh)))
    emergency_total = float(np.sum(emergency_kwh))
    caption = (
        f"图 2：{date.date()} 的因果净负荷中心预测 MAE 为 {mae:.2f} kWh/时段；"
        f"80%风险分位及保形裕度用于形成计划，该日紧急购电总量为"
        f" {emergency_total:.2f} kWh。"
    )

    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.plot(hour, actual_net_kwh, label="实际净负荷", color=COLORS["gray"], linewidth=1.5)
    ax.plot(hour, center_kwh, label="中心预测", color=COLORS["blue"], linewidth=1.4)
    ax.plot(hour, risk_kwh, label="风险调整净负荷", color=COLORS["red"], linewidth=1.3)
    ax.plot(hour, plan_kwh, label="计划购电量", color=COLORS["green"], linewidth=1.3)
    ax.fill_between(
        hour,
        0,
        emergency_kwh,
        label="紧急购电量",
        color=COLORS["orange"],
        alpha=0.35,
    )
    ax.set_title(f"问题2：净负荷预测与购电计划（{date.date()}）")
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("10分钟电量（kWh）")
    ax.set_xlim(0, 24)
    ax.legend(ncol=3, loc="upper center")
    ax.grid(alpha=0.25)
    _save_with_caption(fig, output_path, caption)
    return caption


def plot_monthly_costs(
    daily: pd.DataFrame,
    output_path: Path,
    figure_number: int,
    title: str,
) -> str:
    configure_matplotlib()
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)
    monthly = frame.groupby("month", as_index=False)[
        ["plan_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan"]
    ].sum()
    total = monthly[
        ["plan_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan"]
    ].sum(axis=1)
    max_month = monthly.loc[int(total.idxmax()), "month"]
    max_cost = float(total.max())
    caption = (
        f"图 {figure_number}：月度总费用最高的月份为 {max_month}，"
        f"金额 {max_cost:,.2f} 元。柱体将计划费、调整费和紧急购电费分开，"
        "可直接识别费用增长来自计划规模还是预测缺口。"
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(monthly))
    plan = monthly["plan_cost_yuan"].to_numpy()
    adjustment = monthly["adjustment_cost_yuan"].to_numpy()
    emergency = monthly["emergency_cost_yuan"].to_numpy()
    ax.bar(x, plan, label="计划购电费", color=COLORS["blue"])
    ax.bar(x, adjustment, bottom=plan, label="调整费", color=COLORS["purple"])
    ax.bar(
        x,
        emergency,
        bottom=plan + adjustment,
        label="紧急购电费",
        color=COLORS["orange"],
    )
    ax.set_title(title)
    ax.set_xlabel("月份")
    ax.set_ylabel("费用（元）")
    ax.set_xticks(x, monthly["month"], rotation=35, ha="right")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    _save_with_caption(fig, output_path, caption)
    return caption


def plot_strategy_comparison(
    summary: pd.DataFrame,
    gate_events: pd.DataFrame,
    output_path: Path,
) -> str:
    configure_matplotlib()
    ordered = ["Q2固定价日前", "Q3固定价滚动", "Q4-2波动价日前", "Q4-3波动价滚动"]
    frame = summary.set_index("model").loc[ordered].reset_index()
    q2 = float(frame.loc[frame["model"] == "Q2固定价日前", "total_cost_yuan"].iloc[0])
    q3 = float(frame.loc[frame["model"] == "Q3固定价滚动", "total_cost_yuan"].iloc[0])
    reduction = 100.0 * (q2 - q3) / q2 if q2 else 0.0
    accepted = int(gate_events["accepted"].sum()) if len(gate_events) else 0
    considered = int(len(gate_events))
    caption = (
        f"图 5：固定电价下，滚动方案相对日前方案的总费用变化为"
        f" {reduction:.2f}%（正值表示下降）；日内门控共接受"
        f" {accepted}/{considered} 次候选调整。"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    x = np.arange(len(frame))
    plan = frame["plan_cost_yuan"].to_numpy()
    adjustment = frame["adjustment_cost_yuan"].to_numpy()
    emergency = frame["emergency_cost_yuan"].to_numpy()
    axes[0].bar(x, plan, label="计划购电费", color=COLORS["blue"])
    axes[0].bar(x, adjustment, bottom=plan, label="调整费", color=COLORS["purple"])
    axes[0].bar(
        x,
        emergency,
        bottom=plan + adjustment,
        label="紧急购电费",
        color=COLORS["orange"],
    )
    axes[0].set_title("四种调度策略费用对比")
    axes[0].set_xlabel("模型")
    axes[0].set_ylabel("评价期总费用（元）")
    axes[0].set_xticks(x, frame["model"], rotation=25, ha="right")
    axes[0].legend(loc="upper center")
    axes[0].grid(axis="y", alpha=0.25)

    if len(gate_events):
        gate = gate_events.groupby(["price_case", "release_time"])["accepted"].sum().unstack(0)
        gate.plot(
            kind="bar",
            ax=axes[1],
            color=[COLORS["green"], COLORS["red"]][: len(gate.columns)],
        )
        axes[1].set_title("各发布时间被接受的调整次数")
        axes[1].set_xlabel("预报发布时间")
        axes[1].set_ylabel("接受次数（次）")
        axes[1].legend(title="价格口径")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "无门控记录", ha="center", va="center")
    _save_with_caption(fig, output_path, caption)
    return caption
