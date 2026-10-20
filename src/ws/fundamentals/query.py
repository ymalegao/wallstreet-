"""Point-in-time selectors for annual SEC facts and filing metadata."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime
from math import fabs
from typing import Literal

import polars as pl

ANNUAL_FORMS = ("10-K", "10-K/A")
ANNUAL_DURATION_DAYS = (350, 380)

BOOK_EQUITY_TAGS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
NET_INCOME_TAGS = ("NetIncomeLoss", "ProfitLoss")
OPERATING_CASH_FLOW_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
ASSETS_TAGS = ("Assets",)
REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)
COST_OF_REVENUE_TAGS = ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")
GROSS_PROFIT_TAGS = ("GrossProfit",)
SHARES_TAGS = ("dei:EntityCommonStockSharesOutstanding",)
FLOW_TAG_GROUPS = {
    "net_income": NET_INCOME_TAGS,
    "operating_cash_flow": OPERATING_CASH_FLOW_TAGS,
    "revenue": REVENUE_TAGS,
    "cost_of_revenue": COST_OF_REVENUE_TAGS,
    "gross_profit": GROSS_PROFIT_TAGS,
}
_RELEVANT_TAGS = set(
    BOOK_EQUITY_TAGS
    + NET_INCOME_TAGS
    + OPERATING_CASH_FLOW_TAGS
    + ASSETS_TAGS
    + REVENUE_TAGS
    + COST_OF_REVENUE_TAGS
    + GROSS_PROFIT_TAGS
    + SHARES_TAGS
)


def latest_annual_period_end(facts: pl.DataFrame, cik: str) -> date | None:
    """Return the latest visible period end from an annual filing for one company."""
    annual = facts.filter((pl.col("cik") == cik) & pl.col("form").is_in(ANNUAL_FORMS))
    if annual.is_empty():
        return None
    result = annual.select(pl.col("period_end").max()).item()
    return result if isinstance(result, date) else None


def fundamentals_are_stale(facts: pl.DataFrame, cik: str, as_of: date) -> bool:
    """A company is stale when its latest annual period end is over 18 calendar months old."""
    period_end = latest_annual_period_end(facts, cik)
    if period_end is None:
        return True
    threshold = _subtract_months(as_of, 18)
    return period_end < threshold


def annual_fact(
    facts: pl.DataFrame,
    *,
    cik: str,
    tag: str,
    kind: Literal["flow", "stock"],
    period_end: date | None = None,
) -> dict[str, object] | None:
    """Select the latest visible annual fact, resolving restatements by availability.

    Flow values require a 350–380 day duration from a 10-K. Stock values require an
    exact annual period end and an instant fact from a 10-K. Callers resolve tag
    priorities by invoking this function in priority order.
    """
    candidate = facts.filter(
        (pl.col("cik") == cik)
        & (pl.col("tag") == tag)
        & (pl.col("unit") == "USD")
        & pl.col("form").is_in(ANNUAL_FORMS)
    )
    if kind == "flow":
        candidate = candidate.filter(
            pl.col("period_start").is_not_null()
            & ((pl.col("period_end") - pl.col("period_start")).dt.total_days().is_between(*ANNUAL_DURATION_DAYS))
        )
    else:
        if period_end is None:
            raise ValueError("stock fact selection requires period_end")
        candidate = candidate.filter(
            pl.col("period_start").is_null() & (pl.col("period_end") == period_end)
        )
    if candidate.is_empty():
        return None

    if kind == "flow":
        newest_period_end = candidate.select(pl.col("period_end").max()).item()
        candidate = candidate.filter(pl.col("period_end") == newest_period_end)
    row = candidate.sort(["available_at", "filed", "accession"], nulls_last=True).row(-1, named=True)
    return row


def annual_value(
    facts: pl.DataFrame,
    *,
    cik: str,
    tags: tuple[str, ...],
    kind: Literal["flow", "stock"],
    period_end: date | None = None,
) -> float | None:
    """Return the first visible value in a tag-priority list, without mixing tags."""
    for tag in tags:
        selected = annual_fact(facts, cik=cik, tag=tag, kind=kind, period_end=period_end)
        if selected is not None:
            value = selected["value"]
            return float(value) if value is not None else None
    return None


def latest_filing_shares(facts: pl.DataFrame, cik: str) -> float | None:
    """Sum share-class counts from the latest visible filing accession."""
    candidates = facts.filter(
        (pl.col("cik") == cik)
        & (pl.col("tag") == "dei:EntityCommonStockSharesOutstanding")
        & (pl.col("unit") == "shares")
    )
    if candidates.is_empty():
        return None
    accession = candidates.sort(["available_at", "filed", "accession"], nulls_last=True).row(-1, named=True)[
        "accession"
    ]
    if accession is None:
        return None
    total = candidates.filter(pl.col("accession") == accession).select(pl.col("value").sum()).item()
    return float(total) if total is not None else None


def _subtract_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, month_offset = divmod(month_index, 12)
    month = month_offset + 1
    day = min(value.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


class FundamentalsIndex:
    """Pre-index SEC facts so repeated point-in-time feature rows stay inexpensive."""

    def __init__(self, facts: pl.DataFrame) -> None:
        required = {
            "cik",
            "tag",
            "unit",
            "value",
            "period_start",
            "period_end",
            "form",
            "accession",
            "filed",
            "available_at",
        }
        missing = required - set(facts.columns)
        if missing:
            raise ValueError(f"fundamentals facts are missing columns: {sorted(missing)}")
        self._annual_periods: dict[str, list[dict[str, object]]] = {}
        self._facts: dict[str, dict[str, list[dict[str, object]]]] = {}
        relevant_facts = facts.filter(pl.col("tag").is_in(sorted(_RELEVANT_TAGS)))
        for row in relevant_facts.iter_rows(named=True):
            cik = str(row["cik"]).zfill(10)
            row["cik"] = cik
            self._facts.setdefault(cik, {}).setdefault(str(row["tag"]), []).append(row)
            if row["form"] in ANNUAL_FORMS:
                self._annual_periods.setdefault(cik, []).append(row)
        for by_tag in self._facts.values():
            for rows in by_tag.values():
                rows.sort(key=_fact_order)
        for rows in self._annual_periods.values():
            rows.sort(key=_fact_order)

    def company_values(self, cik: str, as_of: datetime) -> dict[str, float | date | bool | None]:
        """Select current annual values without exposing facts filed after ``as_of``."""
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        cik = str(cik).zfill(10)
        annual = self._visible(self._annual_periods.get(cik, []), as_of)
        latest_end = max((row["period_end"] for row in annual), default=None)
        if not isinstance(latest_end, date):
            return _empty_company_values()
        if latest_end < _subtract_months(as_of.date(), 18):
            return {**_empty_company_values(), "annual_period_end": latest_end, "stale": True}

        selected: dict[str, float | date | bool | None] = {
            "annual_period_end": latest_end,
            "stale": False,
        }
        for name, tags in FLOW_TAG_GROUPS.items():
            selected[name] = self._annual_value(cik, tags, as_of, kind="flow")
        selected["book_equity"] = self._annual_value(
            cik, BOOK_EQUITY_TAGS, as_of, kind="stock", period_end=latest_end
        )
        selected["assets"] = self._annual_value(cik, ASSETS_TAGS, as_of, kind="stock", period_end=latest_end)
        selected["assets_prior"] = self._prior_year_assets(cik, latest_end, as_of)
        selected["shares"] = self._latest_shares(cik, as_of)
        return selected

    def _annual_value(
        self,
        cik: str,
        tags: tuple[str, ...],
        as_of: datetime,
        *,
        kind: Literal["flow", "stock"],
        period_end: date | None = None,
    ) -> float | None:
        for tag in tags:
            candidates = [
                row
                for row in self._visible(self._facts.get(cik, {}).get(tag, []), as_of)
                if row["unit"] == "USD" and row["form"] in ANNUAL_FORMS
            ]
            if kind == "flow":
                candidates = [
                    row
                    for row in candidates
                    if isinstance(row["period_start"], date)
                    and 350 <= (row["period_end"] - row["period_start"]).days <= 380
                ]
                newest_end = max((row["period_end"] for row in candidates), default=None)
                candidates = [row for row in candidates if row["period_end"] == newest_end]
            else:
                candidates = [
                    row
                    for row in candidates
                    if row["period_start"] is None and row["period_end"] == period_end
                ]
            if candidates:
                value = max(candidates, key=_fact_order)["value"]
                return float(value) if value is not None else None
        return None

    def _prior_year_assets(self, cik: str, current_end: date, as_of: datetime) -> float | None:
        rows = [
            row
            for row in self._visible(self._facts.get(cik, {}).get("Assets", []), as_of)
            if row["unit"] == "USD"
            and row["form"] in ANNUAL_FORMS
            and row["period_start"] is None
            and 350 <= (current_end - row["period_end"]).days <= 380
        ]
        if not rows:
            return None
        target_ordinal = current_end.toordinal() - 365
        end_distances = {
            row["period_end"]: fabs(row["period_end"].toordinal() - target_ordinal) for row in rows
        }
        chosen_end = min(end_distances, key=lambda end: (end_distances[end], -end.toordinal()))
        value = max((row for row in rows if row["period_end"] == chosen_end), key=_fact_order)["value"]
        return float(value) if value is not None else None

    def _latest_shares(self, cik: str, as_of: datetime) -> float | None:
        rows = [
            row
            for row in self._visible(self._facts.get(cik, {}).get(SHARES_TAGS[0], []), as_of)
            if row["unit"] == "shares" and row["accession"] is not None
        ]
        if not rows:
            return None
        latest = max(rows, key=_fact_order)
        same_filing = [row for row in rows if row["accession"] == latest["accession"]]
        return sum(float(row["value"]) for row in same_filing if row["value"] is not None)

    @staticmethod
    def _visible(rows: list[dict[str, object]], as_of: datetime) -> list[dict[str, object]]:
        if not rows:
            return []
        stop = bisect_right([row["available_at"] for row in rows], as_of)
        return rows[:stop]


def _fact_order(row: dict[str, object]) -> tuple[datetime, date, str]:
    available_at = row["available_at"]
    filed = row["filed"]
    if not isinstance(available_at, datetime) or not isinstance(filed, date):
        raise ValueError("fundamentals fact has invalid availability or filing date")
    return (available_at, filed, str(row.get("accession") or ""))


def _empty_company_values() -> dict[str, float | date | bool | None]:
    return {
        "annual_period_end": None,
        "stale": True,
        "book_equity": None,
        "net_income": None,
        "operating_cash_flow": None,
        "revenue": None,
        "cost_of_revenue": None,
        "gross_profit": None,
        "assets": None,
        "assets_prior": None,
        "shares": None,
    }
