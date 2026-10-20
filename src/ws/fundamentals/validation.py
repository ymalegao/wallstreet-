"""Ken French factor parsing and characteristic construction diagnostics."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import defaultdict
from datetime import date
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

import polars as pl

from ws.training.metrics import is_month_end

_MONTH = re.compile(r"^\d{6}$")
_CHECKS = {
    "bm": ("bm", 1.0, "HML"),
    "gp_at": ("gp_at", 1.0, "RMW"),
    "asset_growth": ("asset_growth", -1.0, "CMA"),
    "mom_12_1": ("mom_12_1", 1.0, "UMD"),
}


def load_monthly_factors(factor_zip: Path, momentum_zip: Path) -> dict[str, dict[str, float]]:
    """Read monthly HML/RMW/CMA and UMD returns from the specified FF archives."""
    factors = _read_monthly_archive(
        factor_zip,
        required_columns={"HML", "RMW", "CMA"},
    )
    momentum = _read_monthly_archive(momentum_zip, required_columns={"Mom"})
    merged: dict[str, dict[str, float]] = {}
    for month, values in factors.items():
        mom_value = momentum.get(month, {}).get("Mom")
        if mom_value is None:
            continue
        merged[month] = {**values, "UMD": mom_value}
    return merged


def construction_check(frame: pl.DataFrame, factors: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Compare monthly characteristic long-short returns with the related FF factors."""
    required = {"date", "symbol", "raw_forward_return", *(item[0] for item in _CHECKS.values())}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"construction-check frame is missing columns: {sorted(missing)}")
    daily: dict[str, dict[date, list[tuple[float, float, str]]]] = {
        name: defaultdict(list) for name in _CHECKS
    }
    for row in frame.iter_rows(named=True):
        session_date = row["date"]
        forward_return = row["raw_forward_return"]
        if not isinstance(session_date, date) or forward_return is None or not is_month_end(session_date):
            continue
        for name, (feature, direction, _) in _CHECKS.items():
            value = row[feature]
            if value is None:
                continue
            # asset-growth portfolios are bottom minus top by design.
            daily[name][session_date].append((float(value) * direction, float(forward_return), str(row["symbol"])))

    monthly_spreads: dict[str, dict[str, float]] = {name: {} for name in _CHECKS}
    for name, date_groups in daily.items():
        for session_date, observations in date_groups.items():
            if len(observations) < 3:
                continue
            ordered = sorted(observations, key=lambda item: (item[0], item[2]))
            third = max(1, len(ordered) // 3)
            bottom_return = mean(item[1] for item in ordered[:third])
            top_return = mean(item[1] for item in ordered[-third:])
            month = session_date.strftime("%Y-%m")
            monthly_spreads[name][month] = top_return - bottom_return

    output: dict[str, Any] = {"months": sorted(set(factors).intersection(*(set(v) for v in monthly_spreads.values())))}
    output["correlations"] = {}
    output["monthly_spreads"] = {}
    for name, (_, _, factor_name) in _CHECKS.items():
        months = sorted(set(monthly_spreads[name]) & set(factors))
        strategy_values = [monthly_spreads[name][month] for month in months]
        factor_values = [factors[month][factor_name] for month in months]
        output["correlations"][name] = {
            "factor": factor_name,
            "month_count": len(months),
            "pearson_correlation": _correlation(strategy_values, factor_values),
        }
        output["monthly_spreads"][name] = [
            {
                "month": month,
                "long_short_return": monthly_spreads[name][month],
                "factor_return": factors[month][factor_name],
            }
            for month in months
        ]
    return output


def _read_monthly_archive(path: Path, *, required_columns: set[str]) -> dict[str, dict[str, float]]:
    with zipfile.ZipFile(path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"expected one CSV member in {path}, found {csv_members}")
        text = archive.read(csv_members[0]).decode("utf-8-sig", errors="replace")

    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if required_columns.issubset({value.strip() for value in next(csv.reader([line]))})
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"no monthly header with columns {sorted(required_columns)} found in {path}")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    result: dict[str, dict[str, float]] = {}
    for row in reader:
        month_key = str(row.get("") or row.get("Unnamed: 0") or "").strip()
        if not _MONTH.fullmatch(month_key):
            if result:
                break
            continue
        values: dict[str, float] = {}
        for column in required_columns:
            raw = row.get(column)
            try:
                value = float(str(raw).strip()) / 100.0
            except (TypeError, ValueError):
                break
            if value in {-99.99, -999.0}:
                break
            values[column] = value
        else:
            result[f"{month_key[:4]}-{month_key[4:6]}"] = values
    return result


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    return covariance / sqrt(left_ss * right_ss)
