"""C 题微网数据的完整、可复现预处理流水线。

运行方式：
    python src/sxjm/full_preprocessing_pipeline.py

脚本不使用上传目录或机器相关的绝对路径。它会从脚本位置和当前工作目录向上搜索
``data/raw-datta``，优先读取官方 Excel；若 Excel 引擎或文件读取失败，则自动回退到
``data/csv/raw-datta`` 中的 CSV 镜像。
"""

# 启用较新的类型注解语法，同时保持代码结构清晰。
from __future__ import annotations

# hashlib 用于计算文件哈希，保证处理前后可追溯。
import hashlib
# json 用于保存参数、质量检查和运行结果。
import json
# logging 用于输出可读的运行进度和兼容性回退信息。
import logging
# math 用于周期特征中的圆周常数。
import math
# re 用于解析“预报1小时”等字段名和多种时间字符串。
import re
# sys 用于返回规范的进程退出码。
import sys
# traceback 用于在失败时给出完整错误栈。
import traceback
# datetime 和 time 用于兼容 Excel 读取出的日期时间对象。
from datetime import datetime, time
# Path 用于构造跨平台相对路径，不依赖本地上传路径。
from pathlib import Path
# Any 用于质量报告中的通用 JSON 类型注解。
from typing import Any

# 捕获第三方依赖缺失，使环境问题也能给出明确兼容提示。
try:
    # 强制 Matplotlib 使用非交互后端，使脚本可在服务器或无桌面环境运行。
    import matplotlib

    # Agg 后端直接把图形写入 PNG，不需要打开窗口。
    matplotlib.use("Agg")

    # pyplot 用于生成预处理前后对比图。
    import matplotlib.pyplot as plt
    # font_manager 用于自动选择可用中文字体。
    from matplotlib import font_manager
    # NumPy 用于向量计算、插值和数值完整性检查。
    import numpy as np
    # pandas 用于读取、整形、连接和导出表格数据。
    import pandas as pd
except ImportError as exc:
    # 在依赖尚未就绪时给出可直接执行的安装提示。
    print(
        "缺少运行依赖。请先执行 `pip install pandas numpy openpyxl matplotlib`，"
        f"原始错误：{exc}",
        file=sys.stderr,
    )
    # 使用退出码 2 表示运行环境不完整，而不是数据质量失败。
    raise SystemExit(2) from exc


# 设置统一日志格式，便于直接查看自动运行结果。
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# 获取本脚本专用日志器。
LOGGER = logging.getLogger("c_problem_preprocessing")

# 题目规定的调度时间步长为 10 分钟。
STEP_MINUTES = 10
# 将 10 分钟转换为小时，用于 kW 到 kWh 的换算。
STEP_HOURS = STEP_MINUTES / 60.0
# 问题 2 至问题 4 的结果区间从 2025-02-01 开始，因此一月可作为历史特征热身期。
FEATURE_START_DATE = pd.Timestamp("2025-02-01")
# 使用 3 倍 IQR 的极端 Tukey 围栏，只识别非常异常的连续数值，不误删季节峰谷。
IQR_MULTIPLIER = 3.0


class DataPipelineError(RuntimeError):
    """表示输入结构、物理约束或完整性校验无法通过。"""


def find_project_root() -> Path:
    """自动发现包含 data/raw-datta 的项目根目录。"""

    # 将脚本绝对路径的所有父目录作为第一组候选位置。
    script_candidates = [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    # 将当前工作目录及其父目录作为第二组候选位置。
    cwd_candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    # 合并候选目录，并保持脚本附近路径优先。
    candidates = script_candidates + cwd_candidates
    # 用集合避免同一路径被重复检查。
    visited: set[Path] = set()
    # 逐个检查候选根目录。
    for candidate in candidates:
        # 跳过已经检查过的目录。
        if candidate in visited:
            continue
        # 记录该候选目录已经检查。
        visited.add(candidate)
        # 构造官方原始 Excel 数据目录。
        raw_excel_dir = candidate / "data" / "raw-datta"
        # 只要找到该目录，就认为项目根目录识别成功。
        if raw_excel_dir.is_dir():
            return candidate
    # 所有候选目录都失败时，给出明确、可操作的错误信息。
    raise FileNotFoundError(
        "未找到 data/raw-datta。请把脚本放在项目内，或从包含 data 目录的项目根目录运行。"
    )


def read_csv_compatible(path: Path) -> pd.DataFrame:
    """按常见中文数据编码依次尝试读取 CSV。"""

    # 保存每次失败信息，最终错误会包含全部尝试结果。
    errors: list[str] = []
    # UTF-8 BOM、普通 UTF-8 和 GB18030 覆盖常见竞赛数据编码。
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            # 使用 pandas 读取 CSV，并保留原字段名。
            frame = pd.read_csv(path, encoding=encoding)
            # 记录成功使用的编码，便于复现。
            LOGGER.info("CSV 读取成功：%s，编码=%s", path.name, encoding)
            # 返回成功读取的数据。
            return frame
        except Exception as exc:  # noqa: BLE001 - 需要兼容多种底层读取异常。
            # 将失败类型和消息保存，而不是静默忽略。
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    # 所有编码都失败时，抛出包含尝试记录的异常。
    raise DataPipelineError(f"CSV 读取失败：{path}\n" + "\n".join(errors))


def read_table_compatible(
    project_root: Path,
    workbook_name: str,
    sheet_name: str,
    fallback_csv_name: str,
) -> tuple[pd.DataFrame, str]:
    """优先读取 Excel 工作表，失败时回退到对应 CSV。"""

    # 构造官方 Excel 文件位置。
    excel_path = project_root / "data" / "raw-datta" / workbook_name
    # 构造 CSV 镜像位置。
    csv_path = project_root / "data" / "csv" / "raw-datta" / fallback_csv_name
    # 保存 Excel 失败原因，便于回退时说明。
    excel_error: Exception | None = None
    # 只有文件存在时才尝试 Excel 读取。
    if excel_path.is_file():
        try:
            # openpyxl 是 xlsx 的稳定读取引擎，keep_default_na=False 可保留版式空字符串。
            frame = pd.read_excel(
                excel_path,
                sheet_name=sheet_name,
                engine="openpyxl",
                keep_default_na=False,
            )
            # 记录 Excel 读取成功。
            LOGGER.info("Excel 读取成功：%s / %s", workbook_name, sheet_name)
            # 返回数据和来源说明。
            return frame, f"Excel:{workbook_name}/{sheet_name}"
        except Exception as exc:  # noqa: BLE001 - 兼容引擎缺失、文件损坏等情况。
            # 保存异常，稍后写入日志。
            excel_error = exc
            # 明确提示即将回退，不把错误静默吞掉。
            LOGGER.warning(
                "Excel 读取失败，将尝试 CSV：%s / %s；原因=%s: %s",
                workbook_name,
                sheet_name,
                type(exc).__name__,
                exc,
            )
    else:
        # 文件不存在也作为一种可兼容的读取失败。
        excel_error = FileNotFoundError(excel_path)
        # 记录缺少 Excel 的事实。
        LOGGER.warning("未找到 Excel，将尝试 CSV：%s", excel_path)
    # 检查 CSV 镜像是否存在。
    if csv_path.is_file():
        # 使用多编码兼容函数读取 CSV。
        frame = read_csv_compatible(csv_path)
        # 返回数据和来源说明。
        return frame, f"CSV:{fallback_csv_name}"
    # Excel 和 CSV 都无法使用时，给出完整失败信息。
    raise FileNotFoundError(
        f"无法读取 {workbook_name}/{sheet_name}。Excel 错误：{excel_error}；CSV 不存在：{csv_path}"
    )


def sha256(path: Path) -> str:
    """计算文件 SHA-256，用于确认原始数据未被覆盖。"""

    # 创建 SHA-256 哈希对象。
    digest = hashlib.sha256()
    # 以二进制只读方式打开文件。
    with path.open("rb") as handle:
        # 分块读取，避免一次性把大文件载入内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # 将当前块加入哈希计算。
            digest.update(chunk)
    # 返回十六进制哈希字符串。
    return digest.hexdigest()


def time_to_minute(value: Any) -> int:
    """把 Excel 时间、字符串时间和 0:00+1 统一转换为日内分钟数。"""

    # Excel 时间单元格通常由 openpyxl 读取为 datetime.time。
    if isinstance(value, time):
        # 按小时乘 60 加分钟得到日内分钟。
        return value.hour * 60 + value.minute
    # 某些引擎可能把表头时间读取为完整 datetime。
    if isinstance(value, datetime):
        # 只提取时和分。
        return value.hour * 60 + value.minute
    # 某些 Excel 引擎可能把时间读取为一天的浮点比例。
    if isinstance(value, (float, np.floating)) and 0.0 <= float(value) <= 1.0:
        # 将一天比例转换为分钟，并四舍五入到整数分钟。
        return int(round(float(value) * 24 * 60))
    # 其余情况统一转为去除首尾空白的字符串。
    text = str(value).strip()
    # 题面明确规定 0:00+1 表示第二天 0:00，即当日 24:00。
    if text in {"0:00+1", "00:00+1", "24:00", "24:00:00"}:
        # 返回 1440 分钟，保留跨日边界含义。
        return 24 * 60
    # 同时兼容 H:MM、HH:MM 和带秒格式。
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", text)
    # 无法匹配时拒绝猜测，防止时间错位。
    if not match:
        raise DataPipelineError(f"无法解析时间值：{value!r}")
    # 提取小时与分钟。
    hour, minute = map(int, match.groups())
    # 检查普通时刻的合法范围。
    if hour > 23 or minute > 59:
        raise DataPipelineError(f"时间越界：{value!r}")
    # 返回日内分钟数。
    return hour * 60 + minute


def minute_label(minute: int) -> str:
    """把分钟数格式化为统一的 HH:MM 标签。"""

    # 1440 分钟单独显示为 24:00，避免与日初 00:00 混淆。
    if minute == 24 * 60:
        return "24:00"
    # 其余时刻用两位小时和分钟表示。
    return f"{minute // 60:02d}:{minute % 60:02d}"


def validate_ten_minute_grid(minutes: list[int]) -> None:
    """检查时间列是否完整覆盖 144 个 10 分钟区间。"""

    # 构造理论上应出现的 10、20、…、1440 分钟序列。
    expected = list(range(STEP_MINUTES, 24 * 60 + STEP_MINUTES, STEP_MINUTES))
    # 顺序或数量不一致均意味着调度网格无法直接使用。
    if minutes != expected:
        raise DataPipelineError("10 分钟时间网格不完整、重复或顺序错误。")


def numeric_series(series: pd.Series, label: str) -> pd.Series:
    """把连续型字段严格转换为浮点数，并对转换失败给出字段名。"""

    try:
        # errors='raise' 可避免无法解析的文本被悄悄变成缺失值。
        converted = pd.to_numeric(series, errors="raise").astype(float)
        # 返回浮点序列，使整数零和小数功率具有统一类型。
        return converted
    except Exception as exc:  # noqa: BLE001 - 需要包装 pandas 的多种转换异常。
        # 抛出带业务字段名的异常，便于定位原始表。
        raise DataPipelineError(f"字段 {label} 无法转换为连续数值：{exc}") from exc


def interpolate_short_numeric_gaps(
    frame: pd.DataFrame,
    columns: list[str],
    group_columns: list[str] | None,
    max_gap: int,
    audit: dict[str, Any],
) -> pd.DataFrame:
    """仅对连续时间数值的短内部缺口做线性插值，长缺口直接报错。"""

    # 复制数据，避免意外修改调用方持有的原始表。
    result = frame.copy()
    # 逐个连续数值字段处理，不能对日期或类别字段套用数值插值。
    for column in columns:
        # 统计原始缺失数量。
        before_missing = int(result[column].isna().sum())
        # 没有缺失时明确记录“无需填充”，不做多余操作。
        if before_missing == 0:
            audit.setdefault("numeric_missing", {})[column] = {
                "before": 0,
                "filled": 0,
                "remaining": 0,
                "method": "无需填充",
            }
            continue
        # 有分组键时，只允许在同一物理序列内部插值。
        if group_columns:
            # transform 保持原行索引，limit_area='inside' 禁止外推边界。
            result[column] = result.groupby(group_columns, sort=False)[column].transform(
                lambda values: values.interpolate(
                    method="linear",
                    limit=max_gap,
                    limit_area="inside",
                )
            )
        else:
            # 单一连续序列直接进行短内部缺口插值。
            result[column] = result[column].interpolate(
                method="linear",
                limit=max_gap,
                limit_area="inside",
            )
        # 统计插值后仍未解决的缺失。
        after_missing = int(result[column].isna().sum())
        # 写入审计记录，说明方法和实际填充数量。
        audit.setdefault("numeric_missing", {})[column] = {
            "before": before_missing,
            "filled": before_missing - after_missing,
            "remaining": after_missing,
            "method": f"同序列线性插值，最多连续 {max_gap} 个内部时间点",
        }
        # 长缺口、边界缺口或整组缺失不能靠猜测填补。
        if after_missing > 0:
            raise DataPipelineError(
                f"字段 {column} 插值后仍有 {after_missing} 个缺失值；为避免制造数据，流程已停止。"
            )
    # 返回完成安全插值的数据。
    return result


def read_typical_day(project_root: Path, audit: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取附件 1，并生成统一的代表日 10 分钟表。"""

    # 使用 Excel 优先、CSV 回退方式读取附件 1。
    raw, source = read_table_compatible(
        project_root,
        "附件1.xlsx",
        "Sheet1",
        "附件1__Sheet1.csv",
    )
    # 记录实际读取来源。
    audit["input_sources"]["attachment_1"] = source
    # 检查必需字段是否齐全。
    required = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    # 找出缺失字段。
    missing_columns = [column for column in required if column not in raw.columns]
    # 缺字段时无法可靠继续。
    if missing_columns:
        raise DataPipelineError(f"附件 1 缺少字段：{missing_columns}")
    # 解析混合时间类型。
    end_minutes = [time_to_minute(value) for value in raw["时间"]]
    # 验证 144 个时段完整。
    validate_ten_minute_grid(end_minutes)
    # 构造统一、显式的时间索引和连续数值字段。
    clean = pd.DataFrame(
        {
            "interval_index": np.arange(1, len(raw) + 1, dtype=int),
            "interval_start": [minute_label(value - STEP_MINUTES) for value in end_minutes],
            "interval_end": [minute_label(value) for value in end_minutes],
            "interval_start_minute": np.asarray(end_minutes, dtype=int) - STEP_MINUTES,
            "interval_end_minute": np.asarray(end_minutes, dtype=int),
            "duration_hours": STEP_HOURS,
            "electricity_price_yuan_per_kwh": numeric_series(raw["电价"], "附件1.电价"),
            "load_kw": numeric_series(raw["小区负载"], "附件1.小区负载"),
            "pv_forecast_kw": numeric_series(raw["光伏发电预测功率"], "附件1.光伏预测"),
        }
    )
    # 对连续数值字段执行“有缺失才填、无缺失不动”的类型适配逻辑。
    clean = interpolate_short_numeric_gaps(
        clean,
        ["electricity_price_yuan_per_kwh", "load_kw", "pv_forecast_kw"],
        group_columns=None,
        max_gap=2,
        audit=audit,
    )
    # 用矩形积分把负载功率换算为每个 10 分钟区间的电量。
    clean["load_energy_kwh"] = clean["load_kw"] * clean["duration_hours"]
    # 用同一规则把光伏预测功率换算为电量。
    clean["pv_forecast_energy_kwh"] = clean["pv_forecast_kw"] * clean["duration_hours"]
    # 返回原始表和清洗表，前者用于“处理前”绘图。
    return raw, clean


def wide_sheet_to_long(
    raw: pd.DataFrame,
    value_name: str,
    audit: dict[str, Any],
) -> pd.DataFrame:
    """把日期×144 时段的连续数值宽表转换为带完整时间戳的长表。"""

    # 第一列是日期维度，不依赖其具体中文转义形式。
    date_column = raw.columns[0]
    # 将日期严格转换为午夜时间戳。
    dates = pd.to_datetime(raw[date_column], errors="coerce").dt.normalize()
    # 日期不能用均值或众数填充，解析失败时必须停止。
    if dates.isna().any():
        raise DataPipelineError(f"{value_name} 的日期列有 {int(dates.isna().sum())} 个无法解析值。")
    # 日期必须唯一，否则长表主键会发生碰撞。
    if dates.duplicated().any():
        raise DataPipelineError(f"{value_name} 存在重复日期。")
    # 取出全部时段列。
    value_columns = list(raw.columns[1:])
    # 将混合表头时间统一为分钟数。
    end_minutes = [time_to_minute(value) for value in value_columns]
    # 验证每日 144 个时段完整。
    validate_ten_minute_grid(end_minutes)
    # 将所有功率或电价单元格严格转换为浮点数矩阵。
    values = raw[value_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # 将每个日期重复 144 次，对应每日时段。
    operating_dates = np.repeat(dates.to_numpy(dtype="datetime64[ns]"), len(end_minutes))
    # 将 144 个分钟偏移对每个日期重复。
    repeated_minutes = np.tile(np.asarray(end_minutes, dtype=int), len(dates))
    # 计算完整区间结束时间戳；1440 分钟会自然进入次日 00:00。
    interval_end = pd.to_datetime(operating_dates) + pd.to_timedelta(repeated_minutes, unit="m")
    # 构造标准长表。
    long = pd.DataFrame(
        {
            "operating_date": pd.to_datetime(operating_dates),
            "interval_index": np.tile(np.arange(1, len(end_minutes) + 1, dtype=int), len(dates)),
            "interval_start": interval_end - pd.Timedelta(minutes=STEP_MINUTES),
            "interval_end": interval_end,
            "duration_hours": STEP_HOURS,
            value_name: values.reshape(-1),
        }
    )
    # 对连续数值的孤立短缺口允许同一时间序列内插值；当前数据实际不会触发。
    long = interpolate_short_numeric_gaps(
        long,
        [value_name],
        group_columns=None,
        max_gap=2,
        audit=audit,
    )
    # 返回长表。
    return long


def read_annual_data(
    project_root: Path,
    audit: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """读取附件 2 和附件 4，转换并合并为统一 10 分钟表。"""

    # 读取附件 2 的小区负载工作表。
    raw_load, source_load = read_table_compatible(
        project_root,
        "附件2.xlsx",
        "小区负载",
        "附件2__小区负载.csv",
    )
    # 读取附件 2 的光伏实际功率工作表。
    raw_pv, source_pv = read_table_compatible(
        project_root,
        "附件2.xlsx",
        "光伏发电实际功率",
        "附件2__光伏发电实际功率.csv",
    )
    # 读取附件 4 的波动电价工作表。
    raw_price, source_price = read_table_compatible(
        project_root,
        "附件4.xlsx",
        "Sheet1",
        "附件4__Sheet1.csv",
    )
    # 记录三个实际读取来源。
    audit["input_sources"].update(
        {
            "attachment_2_load": source_load,
            "attachment_2_pv": source_pv,
            "attachment_4_price": source_price,
        }
    )
    # 负载是连续时间数值，按通用宽转长函数处理。
    load = wide_sheet_to_long(raw_load, "load_kw", audit)
    # 光伏功率是连续时间数值，按相同时间主键处理。
    pv = wide_sheet_to_long(raw_pv, "pv_actual_kw", audit)
    # 电价也是连续时间数值，但保留元/kWh 的物理单位。
    price = wide_sheet_to_long(raw_price, "electricity_price_yuan_per_kwh", audit)
    # 定义三个数据块必须共享的主键和时间字段。
    keys = ["operating_date", "interval_index", "interval_start", "interval_end", "duration_hours"]
    # 一对一合并负载和光伏；任何重复键都会由 validate='one_to_one' 报错。
    annual = load.merge(pv, on=keys, how="inner", validate="one_to_one")
    # 再以同样方式加入波动电价。
    annual = annual.merge(price, on=keys, how="inner", validate="one_to_one")
    # 计算负载电量，保留原负载功率。
    annual["load_energy_kwh"] = annual["load_kw"] * annual["duration_hours"]
    # 计算光伏实际发电量，保留原光伏功率。
    annual["pv_actual_energy_kwh"] = annual["pv_actual_kw"] * annual["duration_hours"]
    # 按日期和时段排序，保证滞后特征严格遵守时间顺序。
    annual = annual.sort_values(["operating_date", "interval_index"], ignore_index=True)
    # 返回原始宽表集合和合并后的长表。
    return {"load": raw_load, "pv": raw_pv, "price": raw_price}, annual


def read_hourly_forecasts(
    project_root: Path,
    audit: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取附件 3，补全结构性日期并转换为小时预测长表。"""

    # 读取附件 3，优先 Excel、失败时回退 CSV。
    raw, source = read_table_compatible(
        project_root,
        "附件3.xlsx",
        "Sheet1",
        "附件3__Sheet1.csv",
    )
    # 记录实际读取来源。
    audit["input_sources"]["attachment_3"] = source
    # 检查两个标识字段。
    if "日期" not in raw.columns or "预报时刻" not in raw.columns:
        raise DataPipelineError("附件 3 缺少日期或预报时刻字段。")
    # 把空字符串统一视为结构性空白。
    raw_dates = raw["日期"].replace(r"^\s*$", np.nan, regex=True)
    # 统计结构性空白数量。
    structural_blank_count = int(raw_dates.isna().sum())
    # 日期是有序标识，按题面每日四行版式向下补全，不能用众数填充。
    filled_dates = raw_dates.ffill()
    # 首行若为空则无法确定所属日期，必须停止。
    if filled_dates.isna().any():
        raise DataPipelineError("附件 3 首个日期为空，无法按版式向下补全。")
    # 把补全后的日期转为标准时间戳。
    dates = pd.to_datetime(filled_dates, errors="coerce").dt.normalize()
    # 任何解析失败均应阻断流程。
    if dates.isna().any():
        raise DataPipelineError("附件 3 存在无法解析的日期。")
    # 将四个预报时刻转为日内分钟。
    release_minutes = raw["预报时刻"].map(time_to_minute)
    # 构造预测发布时间戳。
    issue_timestamp = dates + pd.to_timedelta(release_minutes, unit="m")
    # 定义题面要求的 24 个小时预测字段。
    forecast_columns = [f"预报{hour}小时" for hour in range(1, 25)]
    # 检查全部预测字段是否存在。
    missing_columns = [column for column in forecast_columns if column not in raw.columns]
    # 缺少任何一个预测步长都无法形成完整 24 小时轨迹。
    if missing_columns:
        raise DataPipelineError(f"附件 3 缺少预测字段：{missing_columns}")
    # 复制预测矩阵，避免修改原表。
    forecast_values = raw[forecast_columns].copy()
    # 每个预测字段都必须是连续数值。
    for column in forecast_columns:
        # 转换失败先变为 NaN，随后按同一发布轨迹的短缺口规则处理。
        forecast_values[column] = pd.to_numeric(forecast_values[column], errors="coerce")
    # 加入发布时间戳作为长表主键的一部分。
    forecast_values.insert(0, "issue_timestamp", issue_timestamp)
    # 将 24 个横向预测字段转换为逐目标时刻的长表。
    hourly = forecast_values.melt(
        id_vars="issue_timestamp",
        var_name="horizon_label",
        value_name="pv_forecast_kw",
    )
    # 从字段名中提取 1 至 24 的整数预测步长。
    hourly["horizon_hour"] = hourly["horizon_label"].str.extract(r"(\d+)", expand=False).astype(int)
    # 按发布时间与预测步长排序，使插值只发生在同一次预测轨迹内部。
    hourly = hourly.sort_values(["issue_timestamp", "horizon_hour"], ignore_index=True)
    # 对同一发布时间内最多两个连续小时的内部数值缺口作线性插值。
    hourly = interpolate_short_numeric_gaps(
        hourly,
        ["pv_forecast_kw"],
        group_columns=["issue_timestamp"],
        max_gap=2,
        audit=audit,
    )
    # 构造每个预测值对应的真实目标时间戳。
    hourly["target_timestamp"] = hourly["issue_timestamp"] + pd.to_timedelta(
        hourly["horizon_hour"], unit="h"
    )
    # 单独保存发布日期，便于按日筛选。
    hourly["issue_date"] = hourly["issue_timestamp"].dt.normalize()
    # 单独保存发布时刻标签，便于 One-Hot 编码和绘图。
    hourly["issue_time"] = hourly["issue_timestamp"].dt.strftime("%H:%M")
    # 验证每日必须恰好有四个发布时刻。
    release_counts = hourly[["issue_date", "issue_time"]].drop_duplicates().groupby("issue_date").size()
    # 不等于四说明日期补全或原始版式异常。
    if not release_counts.eq(4).all():
        raise DataPipelineError("附件 3 并非每天恰好四个发布时刻。")
    # 记录缺失填充方法和数量。
    audit["structural_missing"] = {
        "field": "附件3.日期",
        "before": structural_blank_count,
        "filled": structural_blank_count,
        "remaining": 0,
        "method": "按每日 0:00、6:00、12:00、18:00 四行版式前向填充",
    }
    # 按清晰列顺序返回小时长表。
    hourly = hourly[
        [
            "issue_date",
            "issue_time",
            "issue_timestamp",
            "horizon_hour",
            "target_timestamp",
            "pv_forecast_kw",
        ]
    ]
    # 返回原始宽表和处理后的小时长表。
    return raw, hourly


def align_forecasts_to_ten_minutes(
    hourly: pd.DataFrame,
    annual: pd.DataFrame,
    audit: dict[str, Any],
) -> pd.DataFrame:
    """把每次发布的 24 个小时预测线性插值到 10 分钟调度网格。"""

    # 建立“时间戳 -> 实际光伏功率”的锚点映射。
    actual_at_timestamp = annual.set_index("interval_end")["pv_actual_kw"]
    # 构造 10、20、…、1440 分钟的目标提前量。
    target_minutes = np.arange(STEP_MINUTES, 24 * 60 + STEP_MINUTES, STEP_MINUTES, dtype=int)
    # 将提前分钟转换为小时，用于一维线性插值。
    target_hours = target_minutes / 60.0
    # 保存每次发布对应的 144 行插值结果。
    pieces: list[pd.DataFrame] = []
    # 统计年初唯一的无历史锚点情况。
    assumed_zero_anchor_count = 0
    # 逐次预测发布处理，禁止跨不同发布时间插值。
    for issue_timestamp, group in hourly.groupby("issue_timestamp", sort=True):
        # 按 1 至 24 小时排序。
        group = group.sort_values("horizon_hour")
        # 正常情况下用附件 2 的发布时刻实际光伏功率作为 0 小时点。
        if issue_timestamp in actual_at_timestamp.index:
            # 读取实际锚点。
            anchor = float(actual_at_timestamp.loc[issue_timestamp])
        # 2025 年首日 00:00 没有上一日 24:00 观测，只能使用边界假设。
        elif issue_timestamp == pd.Timestamp("2025-01-01 00:00:00"):
            # 夜间光伏取 0 kW，与随后小时预测的零值一致。
            anchor = 0.0
            # 记录唯一一次边界假设。
            assumed_zero_anchor_count += 1
        else:
            # 其他发布时间缺锚点意味着年度数据不完整，不能静默处理。
            raise DataPipelineError(f"缺少发布时刻实际光伏功率：{issue_timestamp}")
        # 构造已知的 0、1、…、24 小时时间点。
        known_hours = np.arange(0, 25, dtype=float)
        # 将实际锚点和 24 个预测值组成完整已知序列。
        known_values = np.concatenate(([anchor], group["pv_forecast_kw"].to_numpy(dtype=float)))
        # 在相邻已知点之间做线性插值，不外推 24 小时范围之外。
        interpolated = np.interp(target_hours, known_hours, known_values)
        # 用整数分钟构造时间戳，避免浮点小时造成纳秒偏移。
        targets = issue_timestamp + pd.to_timedelta(target_minutes, unit="m")
        # 把本次发布的插值结果保存为结构化表。
        piece = pd.DataFrame(
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
        # 将预测功率换算为每个 10 分钟区间的预测电量。
        piece["pv_forecast_energy_kwh"] = piece["pv_forecast_kw"] * STEP_HOURS
        # 将无序的四类发布时间做 One-Hot 编码，不使用带虚假顺序的标签编码。
        for label in ("00:00", "06:00", "12:00", "18:00"):
            # 列名去掉冒号，便于多数建模框架读取。
            encoded_name = f"issue_{label.replace(':', '')}"
            # 当前发布时刻匹配时取 1，否则取 0。
            piece[encoded_name] = int(issue_timestamp.strftime("%H:%M") == label)
        # 将本次发布加入结果集合。
        pieces.append(piece)
    # 合并所有发布时间的 10 分钟预测。
    result = pd.concat(pieces, ignore_index=True)
    # 记录插值和编码方法。
    audit["forecast_alignment"] = {
        "method": "发布时刻实际值作为 0 小时锚点，相邻小时预测点线性插值到 10 分钟",
        "first_boundary_zero_anchor_count": assumed_zero_anchor_count,
        "categorical_encoding": "issue_time 使用 One-Hot；未使用标签编码",
    }
    # 返回对齐结果。
    return result


def extreme_iqr_outlier_flags(
    frame: pd.DataFrame,
    value_column: str,
) -> pd.Series:
    """使用极端 Tukey 围栏识别连续数值候选异常。"""

    # 计算第一四分位数；IQR 不要求数据服从正态分布。
    q1 = float(frame[value_column].quantile(0.25))
    # 计算第三四分位数。
    q3 = float(frame[value_column].quantile(0.75))
    # 计算四分位距。
    iqr = q3 - q1
    # 极差为零时，所有相同值均不应被误判为异常。
    if iqr == 0:
        return pd.Series(False, index=frame.index)
    # 使用 3 倍 IQR 而非常见的 1.5 倍，以适应负载、光伏和价格的真实季节峰谷。
    lower = q1 - IQR_MULTIPLIER * iqr
    # 计算上围栏。
    upper = q3 + IQR_MULTIPLIER * iqr
    # 返回围栏外候选；该标记不会自动改值或删行。
    return (frame[value_column] < lower) | (frame[value_column] > upper)


def add_anomaly_flags(annual: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    """执行物理边界检查和稳健统计异常检测。"""

    # 复制数据，保留未经修改的物理量。
    result = annual.copy()
    # 负载功率为负不符合本题物理定义。
    negative_load = result["load_kw"] < 0
    # 光伏发电功率为负不符合本题物理定义。
    negative_pv = result["pv_actual_kw"] < 0
    # 非有限数不能用于优化或预测。
    nonfinite = ~np.isfinite(
        result[["load_kw", "pv_actual_kw", "electricity_price_yuan_per_kwh"]].to_numpy(dtype=float)
    )
    # 汇总真正的物理非法记录数。
    physical_invalid_count = int(negative_load.sum() + negative_pv.sum() + nonfinite.sum())
    # 物理非法值不能靠统计平滑掩盖，发现时直接阻断。
    if physical_invalid_count > 0:
        raise DataPipelineError(f"发现 {physical_invalid_count} 个物理非法或非有限数值。")
    # 对负载使用极端 Tukey 围栏检测。
    result["load_robust_outlier"] = extreme_iqr_outlier_flags(result, "load_kw")
    # 对光伏实际功率使用相同的非参数检测。
    result["pv_robust_outlier"] = extreme_iqr_outlier_flags(result, "pv_actual_kw")
    # 对波动电价使用相同检测；即使出现候选尖峰也不会未经确认直接改值。
    result["price_robust_outlier"] = extreme_iqr_outlier_flags(
        result, "electricity_price_yuan_per_kwh"
    )
    # 记录检测结果和不自动修正的依据。
    audit["anomaly_detection"] = {
        "physical_invalid_count": physical_invalid_count,
        "method": f"连续数值采用极端 Tukey 围栏：Q1-{IQR_MULTIPLIER}×IQR 至 Q3+{IQR_MULTIPLIER}×IQR；同时执行物理边界检查",
        "candidate_counts": {
            "load_kw": int(result["load_robust_outlier"].sum()),
            "pv_actual_kw": int(result["pv_robust_outlier"].sum()),
            "electricity_price_yuan_per_kwh": int(result["price_robust_outlier"].sum()),
        },
        "correction_or_deletion": "未执行；候选峰谷仍满足物理边界，可能是真实天气、负载或市场波动",
    }
    # 返回带候选异常标记、但物理值未改变的数据。
    return result


def month_to_season(month: int) -> str:
    """把月份映射为无序季节类别。"""

    # 12、1、2 月归为冬季。
    if month in (12, 1, 2):
        return "winter"
    # 3、4、5 月归为春季。
    if month in (3, 4, 5):
        return "spring"
    # 6、7、8 月归为夏季。
    if month in (6, 7, 8):
        return "summer"
    # 其余 9、10、11 月归为秋季。
    return "autumn"


def build_model_features(annual: pd.DataFrame, audit: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """提取时序、周期、滞后、滚动和分类特征，并执行类型适配缩放。"""

    # 复制年度长表，确保特征工程不覆盖基础物理数据。
    features = annual.copy()
    # 净负荷是购电和储能调度的核心连续特征。
    features["net_load_kw"] = features["load_kw"] - features["pv_actual_kw"]
    # 将净负荷换算为 10 分钟电量。
    features["net_load_energy_kwh"] = features["net_load_kw"] * features["duration_hours"]
    # 从区间结束时刻提取月份。
    features["month"] = features["interval_end"].dt.month.astype(int)
    # 提取星期序号；这里只用于派生，不直接作为有序标签输入模型。
    features["day_of_week"] = features["interval_end"].dt.dayofweek.astype(int)
    # 周末是有明确二元含义的布尔特征，可直接编码为 0/1。
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)
    # 将月份映射为无序季节类别。
    features["season"] = features["month"].map(month_to_season)
    # 计算日内分钟位置；24:00 对应下一日 00:00，因此用模运算回到 0。
    minute_of_day = (
        features["interval_end"].dt.hour * 60 + features["interval_end"].dt.minute
    ).astype(float)
    # 用正弦表达日内周期，避免 23:50 与 00:00 在数值上相距很远。
    features["time_sin"] = np.sin(2 * math.pi * minute_of_day / (24 * 60))
    # 用余弦与正弦共同唯一表示日内相位。
    features["time_cos"] = np.cos(2 * math.pi * minute_of_day / (24 * 60))
    # 提取一年中的日期序号。
    day_of_year = features["interval_end"].dt.dayofyear.astype(float)
    # 生成年周期正弦特征。
    features["year_sin"] = np.sin(2 * math.pi * day_of_year / 365.0)
    # 生成年周期余弦特征。
    features["year_cos"] = np.cos(2 * math.pi * day_of_year / 365.0)
    # 一步滞后表示前 10 分钟负载。
    features["load_lag_10min_kw"] = features["load_kw"].shift(1)
    # 六步滞后表示前 1 小时负载。
    features["load_lag_1h_kw"] = features["load_kw"].shift(6)
    # 144 步滞后表示前一日同一时段负载。
    features["load_lag_1d_kw"] = features["load_kw"].shift(144)
    # 前一日同一时段光伏功率可捕捉天气持续性。
    features["pv_lag_1d_kw"] = features["pv_actual_kw"].shift(144)
    # 前一日同一时段电价可用于波动电价特征。
    features["price_lag_1d_yuan_per_kwh"] = features["electricity_price_yuan_per_kwh"].shift(144)
    # shift(1) 后滚动 6 点，确保 1 小时均值不使用当前或未来信息。
    features["load_rolling_1h_mean_kw"] = features["load_kw"].shift(1).rolling(6).mean()
    # shift(1) 后滚动 144 点，得到无数据泄漏的前 24 小时均值。
    features["load_rolling_24h_mean_kw"] = features["load_kw"].shift(1).rolling(144).mean()
    # 光伏变化率描述当前时段相对前 10 分钟的爬坡速度。
    features["pv_ramp_10min_kw"] = features["pv_actual_kw"].diff()
    # 电价变化量描述市场价格短时跳变。
    features["price_change_10min"] = features["electricity_price_yuan_per_kwh"].diff()
    # 只输出 2 月 1 日以后特征，一月作为 144 步滞后和滚动窗口的热身期。
    features = features.loc[features["operating_date"] >= FEATURE_START_DATE].copy()
    # 检查所有滞后和滚动特征在正式输出期均已完整。
    engineered_columns = [
        "load_lag_10min_kw",
        "load_lag_1h_kw",
        "load_lag_1d_kw",
        "pv_lag_1d_kw",
        "price_lag_1d_yuan_per_kwh",
        "load_rolling_1h_mean_kw",
        "load_rolling_24h_mean_kw",
        "pv_ramp_10min_kw",
        "price_change_10min",
    ]
    # 若仍有缺失，说明历史热身数据不足或时间序列断裂。
    if features[engineered_columns].isna().any().any():
        raise DataPipelineError("正式建模区间的滞后或滚动特征仍有缺失。")
    # 用一月历史数据拟合缩放参数，避免用 2 至 12 月目标期信息拟合尺度。
    calibration = annual.loc[annual["operating_date"] < FEATURE_START_DATE].copy()
    # 一月也需要净负荷以拟合对应标准化参数。
    calibration["net_load_kw"] = calibration["load_kw"] - calibration["pv_actual_kw"]
    # 定义适合 Z-score 的连续变量：负载、净负荷和电价。
    standardize_columns = ["load_kw", "net_load_kw", "electricity_price_yuan_per_kwh"]
    # 保存缩放参数，供预测或反变换使用。
    scaler_parameters: dict[str, Any] = {
        "fit_period": [
            calibration["operating_date"].min().date().isoformat(),
            calibration["operating_date"].max().date().isoformat(),
        ],
        "standardization": {},
        "normalization": {},
    }
    # 逐个字段拟合一月均值和样本标准差。
    for column in standardize_columns:
        # 计算均值。
        mean = float(calibration[column].mean())
        # 计算样本标准差。
        std = float(calibration[column].std(ddof=1))
        # 标准差为零时无法标准化，应明确阻断。
        if std == 0 or not np.isfinite(std):
            raise DataPipelineError(f"字段 {column} 的标准差无效，无法标准化。")
        # 生成 Z-score 特征，原物理量列仍保留。
        features[f"{column}_zscore"] = (features[column] - mean) / std
        # 保存参数和选择依据。
        scaler_parameters["standardization"][column] = {
            "mean": mean,
            "std": std,
            "method": "Z-score",
            "reason": "连续变量量纲和波动幅度不同，标准化仅供距离/梯度敏感模型使用",
        }
    # 光伏功率非负、零值有物理意义且存在明确上下界，适合 Min-Max 归一化。
    pv_min = float(calibration["pv_actual_kw"].min())
    # 计算一月光伏最大值。
    pv_max = float(calibration["pv_actual_kw"].max())
    # 极差为零时无法进行 Min-Max 归一化。
    if pv_max == pv_min:
        raise DataPipelineError("一月光伏功率极差为零，无法归一化。")
    # 生成光伏 [0,1] 基准归一化特征；后续月份超出一月最大值时允许大于 1，不截断。
    features["pv_actual_kw_minmax"] = (features["pv_actual_kw"] - pv_min) / (pv_max - pv_min)
    # 保存 Min-Max 参数和不截断说明。
    scaler_parameters["normalization"]["pv_actual_kw"] = {
        "min": pv_min,
        "max": pv_max,
        "method": "Min-Max",
        "reason": "光伏功率非负且零点有明确物理意义；按历史一月尺度归一化，不裁剪越界值",
    }
    # 将星期转换为字符串类别，明确其无序属性。
    weekday_category = features["day_of_week"].map(lambda value: f"dow_{int(value)}")
    # 对星期和季节做 One-Hot，避免标签编码制造虚假大小关系。
    one_hot = pd.get_dummies(
        pd.DataFrame({"weekday": weekday_category, "season": features["season"]}),
        columns=["weekday", "season"],
        dtype=int,
    )
    # 把 One-Hot 列拼接回特征表。
    features = pd.concat([features, one_hot], axis=1)
    # 删除仅用于派生的季节文本、月份号和星期号，避免模型误用隐含顺序。
    features = features.drop(columns=["season", "month", "day_of_week"])
    # 记录编码和复杂特征提取方法。
    audit["feature_engineering"] = {
        "output_start": FEATURE_START_DATE.date().isoformat(),
        "warmup_period": "2025-01-01 至 2025-01-31",
        "continuous_features": [
            "net_load_kw",
            "power_to_energy",
            "10min/1h/1d lags",
            "1h/24h trailing means",
            "PV ramp",
            "price change",
        ],
        "cyclical_features": ["time_sin", "time_cos", "year_sin", "year_cos"],
        "one_hot_features": sorted(one_hot.columns.tolist()),
        "label_encoding": "未使用；weekday、season 和 issue_time 均无自然等级顺序",
        "scaling_scope": "只写入独立 model_features_10min.csv，不改变基础物理量 CSV",
    }
    # 返回完整特征表和可复用缩放参数。
    return features.reset_index(drop=True), scaler_parameters


def frame_quality(
    frame: pd.DataFrame,
    key_columns: list[str],
    numeric_columns: list[str],
) -> dict[str, Any]:
    """计算通用数据完整性指标。"""

    # 将待检数值列转换为浮点矩阵。
    numeric_matrix = frame[numeric_columns].to_numpy(dtype=float)
    # 汇总行列数、主键、缺失、非有限数和数值范围。
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "duplicate_keys": int(frame.duplicated(key_columns).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric_matrix)).sum()),
        "ranges": {
            column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
            for column in numeric_columns
        },
    }


def choose_font() -> str:
    """自动选择可用中文字体，若没有则安全回退。"""

    # 获取系统中 Matplotlib 可识别的字体名称集合。
    available = {font.name for font in font_manager.fontManager.ttflist}
    # 按优先级尝试常见中文字体。
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN"):
        # 找到即返回。
        if candidate in available:
            return candidate
    # 没有中文字体时使用 DejaVu Sans，并在图中采用英文标签避免乱码。
    return "DejaVu Sans"


def make_before_after_figure(
    output_path: Path,
    raw_typical: pd.DataFrame,
    clean_typical: pd.DataFrame,
    raw_annual: dict[str, pd.DataFrame],
    annual: pd.DataFrame,
    raw_forecast: pd.DataFrame,
    forecast_10min: pd.DataFrame,
) -> None:
    """生成四个附件各自的处理前、处理后双列可视化。"""

    # 选择字体。
    font_name = choose_font()
    # 配置全局字体和负号显示。
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    # 创建 4 行 2 列画布，每行对应一个附件，左侧处理前、右侧处理后。
    figure, axes = plt.subplots(4, 2, figsize=(17, 18), constrained_layout=True)
    # 设置统一白色背景。
    figure.patch.set_facecolor("white")
    # 定义协调、可打印的配色。
    colors = {"load": "#2457A7", "pv": "#E69F00", "price": "#2A9D8F", "accent": "#C44E52"}

    # 附件 1 处理前：按原始行序号展示功率，突出原时间字段尚未标准化。
    axes[0, 0].plot(raw_typical.index + 1, pd.to_numeric(raw_typical["小区负载"]), color=colors["load"], label="Load kW")
    axes[0, 0].plot(raw_typical.index + 1, pd.to_numeric(raw_typical["光伏发电预测功率"]), color=colors["pv"], label="PV forecast kW")
    axes[0, 0].set_title("Attachment 1 before: raw row-indexed power")
    axes[0, 0].set_xlabel("Raw row index")
    axes[0, 0].set_ylabel("Power (kW)")
    axes[0, 0].legend(frameon=False, ncol=2)

    # 附件 1 处理后：展示统一时间轴上的 10 分钟电量。
    axes[0, 1].plot(clean_typical["interval_end_minute"] / 60, clean_typical["load_energy_kwh"], color=colors["load"], label="Load energy")
    axes[0, 1].plot(clean_typical["interval_end_minute"] / 60, clean_typical["pv_forecast_energy_kwh"], color=colors["pv"], label="PV forecast energy")
    axes[0, 1].set_title("Attachment 1 after: aligned 10-minute energy")
    axes[0, 1].set_xlabel("Interval end hour")
    axes[0, 1].set_ylabel("Energy per interval (kWh)")
    axes[0, 1].legend(frameon=False, ncol=2)

    # 附件 2 处理前：原始 365×144 负载宽表热力图。
    raw_load_matrix = raw_annual["load"].iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    heat_load = axes[1, 0].imshow(raw_load_matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[1, 0].set_title("Attachment 2 before: 365 x 144 load matrix")
    axes[1, 0].set_xlabel("10-minute column")
    axes[1, 0].set_ylabel("Day of year")
    figure.colorbar(heat_load, ax=axes[1, 0], shrink=0.8, label="Load (kW)")

    # 附件 2 处理后：按运行日聚合负载与光伏电量，展示统一长表可直接计算。
    daily = annual.groupby("operating_date", as_index=False)[["load_energy_kwh", "pv_actual_energy_kwh"]].sum()
    axes[1, 1].plot(daily["operating_date"], daily["load_energy_kwh"], color=colors["load"], linewidth=1.2, label="Daily load")
    axes[1, 1].plot(daily["operating_date"], daily["pv_actual_energy_kwh"], color=colors["pv"], linewidth=1.2, label="Daily PV")
    axes[1, 1].set_title("Attachment 2 after: daily energy from long table")
    axes[1, 1].set_xlabel("Operating date")
    axes[1, 1].set_ylabel("Daily energy (kWh)")
    axes[1, 1].legend(frameon=False, ncol=2)

    # 选择题面要求展示的日期 2025-06-21 作为预测对齐示例。
    sample_date_text = "2025-6-21"
    # 将附件 3 的结构性日期向下补全后筛选示例日。
    raw_forecast_dates = raw_forecast["日期"].replace(r"^\s*$", np.nan, regex=True).ffill().astype(str)
    sample_raw = raw_forecast.loc[pd.to_datetime(raw_forecast_dates).dt.date == pd.Timestamp("2025-06-21").date()].copy()
    # 附件 3 处理前：四次发布的 24 个小时点。
    for _, row in sample_raw.iterrows():
        raw_values = [float(row[f"预报{hour}小时"]) for hour in range(1, 25)]
        axes[2, 0].plot(range(1, 25), raw_values, linewidth=1.5, label=str(row["预报时刻"]))
    axes[2, 0].set_title(f"Attachment 3 before: 24 hourly points ({sample_date_text})")
    axes[2, 0].set_xlabel("Forecast horizon (hour)")
    axes[2, 0].set_ylabel("PV forecast (kW)")
    axes[2, 0].legend(frameon=False, ncol=4)

    # 附件 3 处理后：同一日期四次发布的 144 个 10 分钟插值点。
    sample_after = forecast_10min.loc[forecast_10min["issue_date"] == pd.Timestamp("2025-06-21")]
    for issue_time, group in sample_after.groupby("issue_time", sort=True):
        axes[2, 1].plot(group["lead_minutes"] / 60, group["pv_forecast_kw"], linewidth=1.5, label=issue_time)
    axes[2, 1].set_title("Attachment 3 after: aligned 10-minute forecasts")
    axes[2, 1].set_xlabel("Forecast horizon (hour)")
    axes[2, 1].set_ylabel("PV forecast (kW)")
    axes[2, 1].legend(frameon=False, ncol=4)

    # 附件 4 处理前：原始 365×144 电价宽表热力图。
    raw_price_matrix = raw_annual["price"].iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    heat_price = axes[3, 0].imshow(raw_price_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[3, 0].set_title("Attachment 4 before: 365 x 144 price matrix")
    axes[3, 0].set_xlabel("10-minute column")
    axes[3, 0].set_ylabel("Day of year")
    figure.colorbar(heat_price, ax=axes[3, 0], shrink=0.8, label="Price (yuan/kWh)")

    # 附件 4 处理后：长表按日聚合最小值、均值和最大值。
    daily_price = annual.groupby("operating_date")["electricity_price_yuan_per_kwh"].agg(["min", "mean", "max"]).reset_index()
    axes[3, 1].fill_between(daily_price["operating_date"], daily_price["min"], daily_price["max"], color="#B7E4C7", alpha=0.65, label="Daily min-max")
    axes[3, 1].plot(daily_price["operating_date"], daily_price["mean"], color=colors["price"], linewidth=1.4, label="Daily mean")
    axes[3, 1].set_title("Attachment 4 after: daily statistics from long table")
    axes[3, 1].set_xlabel("Operating date")
    axes[3, 1].set_ylabel("Price (yuan/kWh)")
    axes[3, 1].legend(frameon=False)

    # 为全部子图添加浅色网格，提升打印可读性。
    for axis in axes.flat:
        axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    # 添加总标题，说明左右列含义。
    figure.suptitle("C Problem Data Preprocessing: Before vs After", fontsize=17, fontweight="bold")
    # 以 220 dpi 导出，兼顾论文插图清晰度和文件大小。
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    # 关闭画布释放内存。
    plt.close(figure)


def make_scaling_figure(output_path: Path, features: pd.DataFrame) -> None:
    """生成连续物理量与缩放特征的处理前后分布对比图。"""

    # 创建两行两列画布。
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # 配置白色背景。
    figure.patch.set_facecolor("white")
    # 原始负载分布。
    axes[0, 0].hist(features["load_kw"], bins=50, color="#2457A7", alpha=0.85)
    axes[0, 0].set_title("Before: load in physical units")
    axes[0, 0].set_xlabel("Load (kW)")
    # 标准化负载分布。
    axes[0, 1].hist(features["load_kw_zscore"], bins=50, color="#5B8FF9", alpha=0.85)
    axes[0, 1].set_title("After: load Z-score")
    axes[0, 1].set_xlabel("Standardized load")
    # 原始光伏分布。
    axes[1, 0].hist(features["pv_actual_kw"], bins=50, color="#E69F00", alpha=0.85)
    axes[1, 0].set_title("Before: PV in physical units")
    axes[1, 0].set_xlabel("PV power (kW)")
    # Min-Max 光伏分布。
    axes[1, 1].hist(features["pv_actual_kw_minmax"], bins=50, color="#F4A261", alpha=0.85)
    axes[1, 1].set_title("After: PV Min-Max using January baseline")
    axes[1, 1].set_xlabel("Normalized PV")
    # 添加浅色网格并去除多余边框。
    for axis in axes.flat:
        axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    # 添加总标题。
    figure.suptitle("Feature Scaling Applied Only to the Model Feature Table", fontsize=15, fontweight="bold")
    # 保存高分辨率 PNG。
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    # 释放图形资源。
    plt.close(figure)


def write_preview(output_path: Path, outputs: dict[str, pd.DataFrame]) -> str:
    """保存并返回每个处理后数据表的前 10 行和后 5 行。"""

    # 用列表累积文本，减少重复文件写入。
    sections: list[str] = []
    # 按输出字典顺序逐表生成预览。
    for name, frame in outputs.items():
        # 写入清晰分隔标题。
        sections.append(f"\n{'=' * 100}\n{name} | shape={frame.shape}\n{'=' * 100}")
        # 写入前 10 行，index=False 避免混淆真实字段。
        sections.append("[HEAD 10]\n" + frame.head(10).to_string(index=False))
        # 写入后 5 行。
        sections.append("[TAIL 5]\n" + frame.tail(5).to_string(index=False))
    # 合并全部预览文本。
    text = "\n".join(sections)
    # 保存 UTF-8 文本，供论文撰写或审计查看。
    output_path.write_text(text, encoding="utf-8")
    # 返回同一文本供控制台展示。
    return text


def validate_exported_csvs(
    output_dir: Path,
    expected: dict[str, tuple[int, list[str], list[str]]],
) -> dict[str, Any]:
    """回读 CSV，验证行数、必需字段、主键、缺失和哈希。"""

    # 保存每个 CSV 的回读结果。
    results: dict[str, Any] = {}
    # 逐个文件执行独立校验。
    for name, (expected_rows, key_columns, required_columns) in expected.items():
        # 构造 CSV 路径。
        path = output_dir / name
        # 文件必须存在且非空。
        if not path.is_file() or path.stat().st_size == 0:
            raise DataPipelineError(f"导出文件不存在或为空：{path}")
        # 以 UTF-8 BOM 兼容方式重新读取。
        frame = pd.read_csv(path, encoding="utf-8-sig")
        # 验证行数。
        if len(frame) != expected_rows:
            raise DataPipelineError(f"{name} 行数错误：期望 {expected_rows}，实际 {len(frame)}")
        # 检查必需字段。
        missing_columns = [column for column in required_columns if column not in frame.columns]
        # 缺字段时立即报错。
        if missing_columns:
            raise DataPipelineError(f"{name} 缺少字段：{missing_columns}")
        # 主键必须唯一。
        duplicate_keys = int(frame.duplicated(key_columns).sum())
        # 发现重复主键时拒绝通过。
        if duplicate_keys:
            raise DataPipelineError(f"{name} 有 {duplicate_keys} 个重复主键。")
        # 统计全部缺失单元格。
        missing_cells = int(frame.isna().sum().sum())
        # 最终基础表和特征表都要求无缺失。
        if missing_cells:
            raise DataPipelineError(f"{name} 回读后有 {missing_cells} 个缺失单元格。")
        # 保存通过状态、尺寸、文件大小和哈希。
        results[name] = {
            "status": "PASS",
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "duplicate_keys": duplicate_keys,
            "missing_cells": missing_cells,
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
    # 返回全部回读结果。
    return results


def main() -> int:
    """执行读取、类型适配、预处理、绘图、导出和完整性校验。"""

    try:
        # 自动定位项目根目录。
        project_root = find_project_root()
        # 构造输出目录。
        output_dir = project_root / "data" / "processed_full"
        # 构造图片目录。
        figure_dir = output_dir / "figures"
        # 创建输出目录，已存在时不报错。
        figure_dir.mkdir(parents=True, exist_ok=True)
        # 初始化方法与结果审计记录。
        audit: dict[str, Any] = {
            "project_root_detection": "自动搜索 data/raw-datta，不使用上传路径或硬编码绝对路径",
            "input_sources": {},
        }
        # 读取并处理附件 1。
        raw_typical, typical = read_typical_day(project_root, audit)
        # 读取并处理附件 2 与附件 4。
        raw_annual, annual = read_annual_data(project_root, audit)
        # 添加异常候选标记，但不改变物理值。
        annual = add_anomaly_flags(annual, audit)
        # 读取并处理附件 3。
        raw_forecast, hourly_forecast = read_hourly_forecasts(project_root, audit)
        # 将小时预测对齐到 10 分钟网格，并 One-Hot 编码发布时刻。
        forecast_10min = align_forecasts_to_ten_minutes(hourly_forecast, annual, audit)
        # 构造独立模型特征表，基础物理量表不做缩放。
        model_features, scaler_parameters = build_model_features(annual, audit)
        # 将输出表集中管理。
        outputs = {
            "typical_day_10min.csv": typical,
            "year_actuals_tariff_10min.csv": annual,
            "pv_forecasts_hourly_long.csv": hourly_forecast,
            "pv_forecasts_10min_linear.csv": forecast_10min,
            "model_features_10min.csv": model_features,
        }
        # 逐表导出为带 UTF-8 BOM 的通用 CSV，方便 Excel 和 Python 同时读取。
        for name, frame in outputs.items():
            # 保留约 10 位有效数字，兼顾精度和文件体积。
            frame.to_csv(
                output_dir / name,
                index=False,
                encoding="utf-8-sig",
                float_format="%.10g",
            )
        # 保存标准化和归一化参数，便于后续一致变换和反变换。
        (output_dir / "scaler_parameters.json").write_text(
            json.dumps(scaler_parameters, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 构造每张表的质量指标。
        quality = {
            "typical_day": frame_quality(
                typical,
                ["interval_index"],
                ["electricity_price_yuan_per_kwh", "load_kw", "pv_forecast_kw"],
            ),
            "annual": frame_quality(
                annual,
                ["operating_date", "interval_index"],
                ["load_kw", "pv_actual_kw", "electricity_price_yuan_per_kwh"],
            ),
            "hourly_forecast": frame_quality(
                hourly_forecast,
                ["issue_timestamp", "horizon_hour"],
                ["pv_forecast_kw"],
            ),
            "forecast_10min": frame_quality(
                forecast_10min,
                ["issue_timestamp", "lead_minutes"],
                ["pv_forecast_kw", "pv_forecast_energy_kwh"],
            ),
            "model_features": frame_quality(
                model_features,
                ["operating_date", "interval_index"],
                ["load_kw", "pv_actual_kw", "net_load_kw"],
            ),
        }
        # 检查插值表在整点上是否严格回到原小时预测。
        hourly_check = hourly_forecast.rename(columns={"pv_forecast_kw": "hourly_kw"})
        # 从 10 分钟表筛选 60 分钟倍数。
        aligned_hours = forecast_10min.loc[forecast_10min["lead_minutes"] % 60 == 0].rename(
            columns={"pv_forecast_kw": "interpolated_kw"}
        )
        # 按发布时间和目标时间一对一连接。
        anchor_check = hourly_check.merge(
            aligned_hours[["issue_timestamp", "target_timestamp", "interpolated_kw"]],
            on=["issue_timestamp", "target_timestamp"],
            how="inner",
            validate="one_to_one",
        )
        # 计算整点最大绝对误差。
        max_anchor_error = float(
            np.max(np.abs(anchor_check["hourly_kw"] - anchor_check["interpolated_kw"]))
        )
        # 整点必须完全一致，否则插值构造错误。
        if max_anchor_error > 1e-9:
            raise DataPipelineError(f"小时预测与 10 分钟插值的整点误差过大：{max_anchor_error}")
        # 保存整点一致性检查。
        audit["forecast_alignment"]["max_hourly_anchor_error_kw"] = max_anchor_error
        # 生成各附件的处理前后双可视化图。
        make_before_after_figure(
            figure_dir / "all_datasets_before_after.png",
            raw_typical,
            typical,
            raw_annual,
            annual,
            raw_forecast,
            forecast_10min,
        )
        # 生成标准化和归一化前后分布图。
        make_scaling_figure(figure_dir / "feature_scaling_before_after.png", model_features)
        # 生成全部处理后表的前 10 行和后 5 行预览。
        preview_text = write_preview(output_dir / "processed_data_preview.txt", outputs)
        # 定义最终回读校验规则。
        expected = {
            "typical_day_10min.csv": (
                144,
                ["interval_index"],
                ["interval_index", "load_kw", "pv_forecast_kw"],
            ),
            "year_actuals_tariff_10min.csv": (
                365 * 144,
                ["operating_date", "interval_index"],
                ["operating_date", "load_kw", "pv_actual_kw", "electricity_price_yuan_per_kwh"],
            ),
            "pv_forecasts_hourly_long.csv": (
                365 * 4 * 24,
                ["issue_timestamp", "horizon_hour"],
                ["issue_timestamp", "horizon_hour", "pv_forecast_kw"],
            ),
            "pv_forecasts_10min_linear.csv": (
                365 * 4 * 144,
                ["issue_timestamp", "lead_minutes"],
                ["issue_timestamp", "lead_minutes", "pv_forecast_kw"],
            ),
            "model_features_10min.csv": (
                334 * 144,
                ["operating_date", "interval_index"],
                ["operating_date", "net_load_kw", "time_sin", "time_cos"],
            ),
        }
        # 回读全部 CSV 并执行最终完整性验证。
        export_validation = validate_exported_csvs(output_dir, expected)
        # 保存完整运行报告。
        run_report = {
            "status": "PASS",
            "method_audit": audit,
            "data_quality": quality,
            "export_validation": export_validation,
            "figures": [
                str((figure_dir / "all_datasets_before_after.png").relative_to(project_root)),
                str((figure_dir / "feature_scaling_before_after.png").relative_to(project_root)),
            ],
        }
        # 将运行报告保存为 JSON。
        (output_dir / "preprocessing_run_report.json").write_text(
            json.dumps(run_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 输出总体成功信息。
        LOGGER.info("预处理与完整性校验全部通过，输出目录：%s", output_dir)
        # 在控制台展示所有处理后数据的前 10 行和后 5 行。
        print(preview_text)
        # 在预览后输出机器可读摘要。
        print("\nRUN_SUMMARY")
        # 打印简化运行摘要。
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output_rows": {name: len(frame) for name, frame in outputs.items()},
                    "anomaly_candidates": audit["anomaly_detection"]["candidate_counts"],
                    "max_hourly_anchor_error_kw": max_anchor_error,
                    "output_directory": str(output_dir.relative_to(project_root)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        # 返回 0 表示成功。
        return 0
    except Exception as exc:  # noqa: BLE001 - 顶层必须捕获并报告所有运行失败。
        # 输出简洁错误信息。
        LOGGER.error("预处理失败：%s: %s", type(exc).__name__, exc)
        # 输出完整错误栈，便于定位读取兼容或数据质量问题。
        traceback.print_exc()
        # 返回非零退出码，便于自动化系统识别失败。
        return 1


# 只有直接运行脚本时才启动流水线；被导入时不会自动改写数据。
if __name__ == "__main__":
    # 把 main 的返回值传给操作系统作为进程退出码。
    sys.exit(main())
