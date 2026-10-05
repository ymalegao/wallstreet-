"""SEC EDGAR: 8-K and Form 4 filings with their acceptance timestamps.

Point-in-time rule: an EDGAR filing becomes public at its *acceptance* datetime, not its filing
date. The timezone of ``acceptanceDateTime`` in the submissions JSON must be verified by
``scripts/probe_apis.py`` (it cross-checks the ``ACCEPTANCE-DATETIME`` header, which EDGAR states
in Eastern time). Until it is verified, ``configs/sources.yaml`` keeps it as UNVERIFIED and the
backfill refuses to run.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal

from ws.ingest.http import RateLimitedClient
from ws.ingest.text import html_to_text
from ws.schema import Event, TsOrigin
from ws.timeutil import ET, parse_naive_as, parse_rfc3339, utcnow

SOURCE = "edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
FORMS = ("8-K", "8-K/A", "4")

AcceptanceTz = Literal["utc", "et"]

ITEMS_8K = {
    "1.01": "Entry into a material definitive agreement",
    "1.02": "Termination of a material definitive agreement",
    "1.03": "Bankruptcy or receivership",
    "1.05": "Material cybersecurity incident",
    "2.01": "Completion of acquisition or disposition of assets",
    "2.02": "Results of operations and financial condition",
    "2.03": "Creation of a direct financial obligation",
    "2.04": "Triggering events that accelerate a financial obligation",
    "2.05": "Costs associated with exit or disposal activities",
    "2.06": "Material impairments",
    "3.01": "Notice of delisting or failure to satisfy listing rule",
    "3.02": "Unregistered sales of equity securities",
    "3.03": "Material modification to rights of security holders",
    "4.01": "Change in registrant's certifying accountant",
    "4.02": "Non-reliance on previously issued financial statements",
    "5.01": "Change in control of registrant",
    "5.02": "Departure or appointment of directors or officers",
    "5.03": "Amendments to articles of incorporation or bylaws",
    "5.07": "Submission of matters to a vote of security holders",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
    "9.01": "Financial statements and exhibits",
}


def make_client(user_agent: str) -> RateLimitedClient:
    # SEC fair-access policy: <= 10 requests/second with a descriptive User-Agent.
    if "@" not in user_agent or len(user_agent.split()) < 2:
        raise ValueError("SEC_USER_AGENT must identify the application and a monitored contact email")
    return RateLimitedClient(headers={"User-Agent": user_agent}, max_per_sec=2.0, budget="sec")


def ticker_to_cik(client: RateLimitedClient) -> dict[str, int]:
    """Current ticker -> CIK map. NOT point-in-time: renamed/delisted tickers need a separate history."""
    data = client.get_json(TICKERS_URL)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def _columns_to_rows(cols: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(cols)
    for values in zip(*(cols[k] for k in keys), strict=True):
        yield dict(zip(keys, values, strict=True))


def iter_filings(client: RateLimitedClient, cik: int, forms: tuple[str, ...] = FORMS) -> Iterator[dict[str, Any]]:
    """All filings of the given forms for a CIK (recent block plus paginated history files)."""
    root = client.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
    blocks = [root["filings"]["recent"]]
    for f in root["filings"].get("files", []):
        blocks.append(client.get_json(SUBMISSIONS_URL.format(name=f["name"])))
    for block in blocks:
        for row in _columns_to_rows(block):
            if row.get("form") in forms:
                row["cik"] = cik
                row["tickers"] = root.get("tickers", [])
                yield row


def parse_acceptance(s: str, tz: AcceptanceTz) -> datetime:
    return parse_rfc3339(s) if tz == "utc" else parse_naive_as(s, ET)


def filing_url(row: dict[str, Any]) -> str:
    acc = row["accessionNumber"]
    return ARCHIVE_URL.format(cik=row["cik"], acc_nodash=acc.replace("-", ""), doc=row["primaryDocument"])


def fetch_text(client: RateLimitedClient, row: dict[str, Any], max_chars: int = 20_000) -> str:
    return html_to_text(client.get(filing_url(row)).text)[:max_chars]


def normalize(row: dict[str, Any], *, acceptance_tz: AcceptanceTz, body: str = "") -> Event:
    accepted = parse_acceptance(row["acceptanceDateTime"], acceptance_tz)
    items = [i.strip() for i in (row.get("items") or "").split(",") if i.strip()]
    form = row["form"]
    if form.startswith("8-K"):
        described = "; ".join(f"Item {i}: {ITEMS_8K.get(i, 'unknown item')}" for i in items)
        headline = f"{form} filed: {described}" if described else f"{form} filed"
    else:
        headline = f"Form {form}: insider transaction report"
    return Event(
        event_id=f"{SOURCE}:{row['accessionNumber']}",
        source=SOURCE,
        source_id=row["accessionNumber"],
        first_seen_ts=accepted,
        ts_origin=TsOrigin.VENDOR,
        published_ts=accepted,
        ingested_at=utcnow(),
        tickers=list(row.get("tickers") or []),
        headline=headline,
        body=body,
        url=filing_url(row),
        kind=form,
        meta={"cik": str(row["cik"]), "items": ",".join(items), "filing_date": row.get("filingDate", "")},
    )
