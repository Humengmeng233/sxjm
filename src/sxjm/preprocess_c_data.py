from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw-datta"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "data" / "processed" / "data_quality_report.json"

STEP_MINUTES = 10
STEP_HOURS = STEP_MINUTES / 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_to_minute(value: object) -> int:
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, (float, np.floating)) and 0 <= float(value) <= 1:
        return int(round(float(value) * 24 * 60))
    text = str(value).strip()
    if text in {"0:00+1", "00:00+1", "24:00", "24:00:00"}:
        return 24 * 60
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", text)
    if not match:
        raise ValueError(f"无法解析时间值: {value!r}")
    hour, minute = map(int, match.groups())
    result = hour * 60 + minute
    if hour > 23 or minute > 59:
        raise ValueError(f"时间越界: {value!r}")
    return result


def minute_label(minute: int) -> str:
    if minute == 24 * 60:
        return "24:00"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def validate_grid(minutes: list[int]) -> None:
    expected = list(range(STEP_MINUTES, 24 * 60 + STEP_MINUTES, STEP_MINUTES))
    if minutes != expected:
        raise ValueError("10 分钟时间网格不完整或顺序不正确")


def read_typical_day() -> pd.DataFrame:
    source = pd.read_excel(RAW_DIR / "附件1.xlsx", sheet_name="Sheet1")
    end_minutes = [time_to_minute(value) for value in source["时间"]]
    validate_grid(end_minutes)

    result = pd.DataFrame(
        {
            "interval_index": np.arange(1, len(source) + 1, dtype=int),
            "interval_start": [minute_label(value - STEP_MINUTES) for value in end_minutes],
            "interval_end": [minute_label(value) for value in end_minutes],
            "interval_start_minute": np.asarray(end_minutes, dtype=int) - STEP_MINUTES,
            "interval_end_minute": np.asarray(end_minutes, dtype=int),
            "duration_hours": STEP_HOURS,
            "electricity_price_yuan_per_kwh": pd.to_numeric(source["电价"], errors="raise"),
            "load_kw": pd.to_numeric(source["小区负载"], errors="raise"),
            "pv_forecast_kw": pd.to_numeric(source["光伏发电预测功率"], errors="raise"),
        }
    )
    result["load_energy_kwh"] = result["load_kw"] * result["duration_hours"]
    result["pv_forecast_energy_kwh"] = result["pv_forecast_kw"] * result["duration_hours"]
    return result


def wide_sheet_to_long(path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name=sheet_name)
    date_column = source.columns[0]
    dates = pd.to_datetime(source[date_column], errors="raise").dt.normalize()
    value_columns = list(source.columns[1:])
    end_minutes = [time_to_minute(value) for value in value_columns]
    validate_grid(end_minutes)

    values = source[value_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    operating_dates = np.repeat(dates.to_numpy(dtype="datetime64[ns]"), len(end_minutes))
    repeated_minutes = np.tile(np.asarray(end_minutes, dtype=int), len(dates))
    interval_end = pd.to_datetime(operating_dates) + pd.to_timedelta(repeated_minutes, unit="m")

    result = pd.DataFrame(
        {
            "operating_date": pd.to_datetime(operating_dates),
            "interval_index": np.tile(np.arange(1, len(end_minutes) + 1, dtype=int), len(dates)),
            "interval_start": interval_end - pd.Timedelta(minutes=STEP_MINUTES),
            "interval_end": interval_end,
            "duration_hours": STEP_HOURS,
            value_name: values.reshape(-1),
        }
    )
    return result


def read_year_actuals_and_tariff() -> pd.DataFrame:
    load = wide_sheet_to_long(RAW_DIR / "附件2.xlsx", "小区负载", "load_kw")
    pv = wide_sheet_to_long(RAW_DIR / "附件2.xlsx", "光伏发电实际功率", "pv_actual_kw")
    tariff = wide_sheet_to_long(
        RAW_DIR / "附件4.xlsx", "Sheet1", "electricity_price_yuan_per_kwh"
    )
    keys = ["operating_date", "interval_index", "interval_start", "interval_end", "duration_hours"]
    result = load.merge(pv, on=keys, how="inner", validate="one_to_one")
    result = result.merge(tariff, on=keys, how="inner", validate="one_to_one")
    result["load_energy_kwh"] = result["load_kw"] * result["duration_hours"]
    result["pv_actual_energy_kwh"] = result["pv_actual_kw"] * result["duration_hours"]
    return result.sort_values(["operating_date", "interval_index"], ignore_index=True)


def read_hourly_forecasts() -> pd.DataFrame:
    source = pd.read_excel(RAW_DIR / "附件3.xlsx", sheet_name="Sheet1", keep_default_na=False)
    dates = source["日期"].replace("", np.nan).ffill()
    if dates.isna().any():
        raise ValueError("附件 3 的首个日期为空，无法前向补全")
    dates = pd.to_datetime(dates, errors="raise").dt.normalize()
    release_minutes = source["预报时刻"].map(time_to_minute)
    issue_timestamp = dates + pd.to_timedelta(release_minutes, unit="m")

    forecast_columns = [f"预报{h}小时" for h in range(1, 25)]
    long = source[forecast_columns].copy()
    long.insert(0, "issue_timestamp", issue_timestamp)
    long = long.melt(
        id_vars="issue_timestamp", var_name="horizon_label", value_name="pv_forecast_kw"
    )
    long["horizon_hour"] = long["horizon_label"].str.extract(r"(\d+)", expand=False).astype(int)
    long["pv_forecast_kw"] = pd.to_numeric(long["pv_forecast_kw"], errors="raise")
    long["target_timestamp"] = long["issue_timestamp"] + pd.to_timedelta(
        long["horizon_hour"], unit="h"
    )
    long["issue_date"] = long["issue_timestamp"].dt.normalize()
    long["issue_time"] = long["issue_timestamp"].dt.strftime("%H:%M")
    return long[
        [
            "issue_date",
            "issue_time",
            "issue_timestamp",
            "horizon_hour",
            "target_timestamp",
            "pv_forecast_kw",
        ]
    ].sort_values(["issue_timestamp", "horizon_hour"], ignore_index=True)


def align_forecasts_to_ten_minutes(
    hourly: pd.DataFrame, actuals: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    actual_at_timestamp = actuals.set_index("interval_end")["pv_actual_kw"]
    target_minutes = np.arange(STEP_MINUTES, 24 * 60 + STEP_MINUTES, STEP_MINUTES, dtype=int)
    target_hours = target_minutes / 60
    pieces: list[pd.DataFrame] = []
    assumed_zero_anchors = 0

    for issue_timestamp, group in hourly.groupby("issue_timestamp", sort=True):
        group = group.sort_values("horizon_hour")
        if issue_timestamp in actual_at_timestamp.index:
            anchor = float(actual_at_timestamp.loc[issue_timestamp])
        elif issue_timestamp == pd.Timestamp("2025-01-01 00:00:00"):
            anchor = 0.0
            assumed_zero_anchors += 1
        else:
            raise ValueError(f"缺少发布时刻实际功率: {issue_timestamp}")

        known_hours = np.arange(0, 25, dtype=float)
        known_values = np.concatenate(([anchor], group["pv_forecast_kw"].to_numpy(dtype=float)))
        interpolated = np.interp(target_hours, known_hours, known_values)
        targets = issue_timestamp + pd.to_timedelta(target_minutes, unit="m")
        pieces.append(
            pd.DataFrame(
                {
                    "issue_date": issue_timestamp.normalize(),
                    "issue_time": issue_timestamp.strftime("%H:%M"),
                    "issue_timestamp": issue_timestamp,
                    "target_timestamp": targets,
                    "lead_minutes": target_minutes,
                    "pv_forecast_kw": interpolated,
                    "anchor_pv_actual_kw": anchor,
                }
            )
        )

    return pd.concat(pieces, ignore_index=True), assumed_zero_anchors


def check_frame(frame: pd.DataFrame, key_columns: list[str], value_columns: list[str]) -> dict[str, object]:
    values = frame[value_columns].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_keys": int(frame.duplicated(key_columns).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "nonfinite_values": int((~np.isfinite(values)).sum()),
        "negative_values": int((values < 0).sum()),
        "value_ranges": {
            column: {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
            for column in value_columns
        },
    }


def json_ready(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"无法序列化 {type(value).__name__}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    typical = read_typical_day()
    actuals = read_year_actuals_and_tariff()
    hourly = read_hourly_forecasts()
    forecast_10min, assumed_zero_anchors = align_forecasts_to_ten_minutes(hourly, actuals)
    annual_means = actuals.groupby("interval_index", as_index=False)[
        ["electricity_price_yuan_per_kwh", "load_kw", "pv_actual_kw"]
    ].mean()
    typical_consistency = {
        "electricity_price_max_abs_difference_from_annual_interval_mean": float(
            np.max(
                np.abs(
                    typical["electricity_price_yuan_per_kwh"].to_numpy()
                    - annual_means["electricity_price_yuan_per_kwh"].to_numpy()
                )
            )
        ),
        "load_max_abs_difference_from_annual_interval_mean_kw": float(
            np.max(np.abs(typical["load_kw"].to_numpy() - annual_means["load_kw"].to_numpy()))
        ),
        "pv_max_abs_difference_from_annual_interval_mean_kw": float(
            np.max(
                np.abs(typical["pv_forecast_kw"].to_numpy() - annual_means["pv_actual_kw"].to_numpy())
            )
        ),
    }

    outputs = {
        "typical_day_10min.csv": typical,
        "year_actuals_tariff_10min.csv": actuals,
        "pv_forecasts_hourly_long.csv": hourly,
        "pv_forecasts_10min_linear.csv": forecast_10min,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig", float_format="%.10g")

    parameters = {
        "provided": {
            "storage_nameplate_capacity_kwh": 12000,
            "storage_operating_min_kwh": 1200,
            "storage_operating_max_kwh": 10800,
            "storage_max_charge_discharge_power_kw": 5000,
            "storage_initial_energy_2025_01_01_00_00_kwh": 6000,
            "charge_discharge_efficiency": 0.90,
            "emergency_purchase_price_multiplier": 5.0,
            "down_adjustment_breach_price_multiplier": 0.50,
            "up_adjustment_increment_price_multiplier": 1.50,
        },
        "derived": {
            "dispatch_interval_minutes": STEP_MINUTES,
            "dispatch_interval_hours": STEP_HOURS,
            "max_charge_discharge_energy_per_interval_kwh": 5000 * STEP_HOURS,
        },
        "modeling_assumptions_not_applied_to_source_values": {
            "efficiency_convention": "充电效率与放电效率分别取0.90；题面使用‘充放电效率为90%’，按双向过程同一效率解释。",
            "state_continuity": "问题2至4逐日滚动时，前一日24:00储电量作为次日0:00储电量；仅问题1强制日初日末相等。",
            "simultaneous_charge_discharge": "同一10分钟时段不允许同时充电和放电，符合单一储能设备物理逻辑。",
            "grid_export": "题面只定义购电及其费用，未定义售电收益，故外网购电量不取负值；光伏富余可弃光或充电。",
        },
    }
    (OUTPUT_DIR / "model_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    quality = {
        "source_scope": {
            "raw_workbooks": [path.name for path in sorted(RAW_DIR.glob("附件*.xlsx"))],
            "excluded_from_inputs": "data/result-data 与 data/csv/result-data 为结果模板或既有结果，不作为原始输入。",
            "sha256": {path.name: sha256(path) for path in sorted(RAW_DIR.glob("附件*.xlsx"))},
        },
        "preprocessing_decision": "需要结构性预处理；不需要删行、异常值替换或数值插补。",
        "transformations": [
            "统一附件1混合时间类型，并把0:00+1标准化为24:00。",
            "将附件2和附件4的365×144宽表转为10分钟长表并按键一一合并。",
            "将附件3每日空白日期按四个发布时刻的版式向下补全，再把24个预测字段转成长表。",
            "按E=P×(10/60)生成负载和光伏电量字段，原始功率值保持不变。",
            "用发布时刻实际光伏功率作为0小时锚点，对小时预测作线性插值，生成10分钟对齐版本；同时保留未插值的小时长表。",
        ],
        "assumptions_applied_in_preprocessing": {
            "time_semantics": "00:10至24:00均解释为前一个10分钟区间的结束时刻。",
            "power_to_energy": "每个10分钟内功率按区间常值处理，因此kWh=kW×1/6小时。",
            "forecast_interpolation": "0小时采用发布时刻实际光伏功率，1至24小时采用附件3预测点，点间线性插值。",
            "first_anchor": "2025-01-01 00:00无上一日24:00观测，取0 kW；该时刻为夜间，且其随后小时预测为0。",
        },
        "checks": {
            "typical_day": check_frame(
                typical,
                ["interval_index"],
                ["electricity_price_yuan_per_kwh", "load_kw", "pv_forecast_kw"],
            ),
            "year_actuals_tariff": check_frame(
                actuals,
                ["operating_date", "interval_index"],
                ["load_kw", "pv_actual_kw", "electricity_price_yuan_per_kwh"],
            ),
            "hourly_forecasts": check_frame(
                hourly,
                ["issue_timestamp", "horizon_hour"],
                ["pv_forecast_kw"],
            ),
            "ten_minute_forecasts": check_frame(
                forecast_10min,
                ["issue_timestamp", "lead_minutes"],
                ["pv_forecast_kw", "anchor_pv_actual_kw"],
            ),
            "assumed_zero_anchor_count": assumed_zero_anchors,
            "actuals_date_range": [
                actuals["operating_date"].min().date().isoformat(),
                actuals["operating_date"].max().date().isoformat(),
            ],
            "actuals_unique_operating_dates": int(actuals["operating_date"].nunique()),
            "forecast_issue_date_range": [
                hourly["issue_date"].min().date().isoformat(),
                hourly["issue_date"].max().date().isoformat(),
            ],
            "forecast_unique_issue_dates": int(hourly["issue_date"].nunique()),
            "forecast_release_times": sorted(hourly["issue_time"].unique().tolist()),
            "typical_day_cross_source_consistency": typical_consistency,
        },
        "outputs": {name: {"rows": int(len(frame)), "columns": int(frame.shape[1])} for name, frame in outputs.items()},
    }
    REPORT_PATH.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, default=json_ready), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2, default=json_ready))


if __name__ == "__main__":
    main()
