from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Optional

import pandas as pd
import requests
import urllib3
import yfinance as yf
from langchain_core.tools import tool


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
)


def _summarize_exception(exc: Exception) -> str:
    text = str(exc).split(" (Caused by", 1)[0].splitlines()[0]
    return f"{type(exc).__name__}: {text}"


def _session_get(url: str, **kwargs) -> requests.Response:
    try:
        return SESSION.get(url, **kwargs)
    except requests.exceptions.SSLError:
        # Fallback for hosts with incomplete certificate chains in some environments.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
            return SESSION.get(url, verify=False, **kwargs)


def _parse_trade_date(curr_date: str) -> datetime:
    return datetime.strptime(curr_date, "%Y-%m-%d")


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "N/A"}:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "N/A"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _fmt_int(value: Any) -> str:
    parsed = _to_int(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:,}"


def _fmt_float(value: Any, digits: int = 2) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.{digits}f}"


def _fetch_html(url: str, curr_date: str) -> str:
    trade_date = _parse_trade_date(curr_date)
    params_candidates = [
        {"queryDate": trade_date.strftime("%Y/%m/%d")},
        {"queryDate": trade_date.strftime("%Y%m%d")},
        {"queryType": "1", "goDay": trade_date.strftime("%Y/%m/%d")},
        {"queryType": "1", "goDay": trade_date.strftime("%Y%m%d")},
        {"date": trade_date.strftime("%Y%m%d")},
        {"date": trade_date.strftime("%Y/%m/%d")},
        None,
    ]

    last_html = ""
    for params in params_candidates:
        response = _session_get(url, params=params, timeout=30)
        response.raise_for_status()
        last_html = response.text
        if "taifex.com.tw" in url and params and "queryDate" in params:
            return last_html
        if trade_date.strftime("%Y/%m/%d") in last_html or trade_date.strftime("%Y%m%d") in last_html:
            return last_html

    return last_html


def _fetch_html_with_fallback(urls: list[str], curr_date: str) -> tuple[str, str]:
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            return _fetch_html(url, curr_date), url
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
    raise RuntimeError(f"Unable to fetch HTML from all candidates: {urls}") from last_error


def _read_taifex_table(url: str, curr_date: str, table_index: int = 0) -> pd.DataFrame:
    html = _fetch_html(url, curr_date)
    tables = pd.read_html(StringIO(html))
    if table_index >= len(tables):
        raise ValueError(f"No table found at index {table_index} for {url}")
    return tables[table_index]


def _read_taifex_table_with_fallback(
    urls: list[str], curr_date: str, table_index: int = 0
) -> tuple[pd.DataFrame, str]:
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            return _read_taifex_table(url, curr_date, table_index), url
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
    raise RuntimeError(
        f"No TAIFEX table found at index {table_index} for URL candidates: {urls}"
    ) from last_error


def _find_price_index_row(data: list[list[Any]], label: str) -> Optional[list[Any]]:
    for row in data:
        if str(row[0]).strip() == label:
            return row
    return None


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).strip().lower()


def _find_row_by_aliases(data: list[list[Any]], aliases: list[str]) -> Optional[list[Any]]:
    normalized_aliases = [_normalize_text(alias) for alias in aliases]
    for row in data:
        row_label = _normalize_text(row[0]) if row else ""
        for alias in normalized_aliases:
            if alias and (row_label == alias or alias in row_label):
                return row
    return None


def _safe_row_value(row: Optional[list[Any]], index: int) -> Any:
    if row is None:
        return None
    if index < 0 or index >= len(row):
        return None
    return row[index]


def _series_from_yfinance(symbol: str, curr_date: str, look_back_days: int = 10):
    trade_date = _parse_trade_date(curr_date)
    start_date = (trade_date - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    end_date = (trade_date + timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        history = yf.Ticker(symbol).history(start=start_date, end=end_date)
    except Exception:  # pragma: no cover - network path
        return None
    if history.empty:
        return None

    history = history.copy()
    history.index = pd.to_datetime(history.index)
    history["_date"] = history.index.tz_localize(None).normalize()
    target_date = pd.Timestamp(trade_date.date())
    eligible = history[history["_date"] <= target_date]
    if eligible.empty:
        eligible = history
    return eligible.iloc[-1]


def _market_label(vix: Optional[float], breadth_ratio: Optional[float], foreign_net: Optional[int]) -> str:
    score = 0
    if vix is not None:
        if vix <= 18:
            score += 1
        elif vix >= 25:
            score -= 1
    if breadth_ratio is not None:
        if breadth_ratio >= 1.3:
            score += 1
        elif breadth_ratio <= 0.8:
            score -= 1
    if foreign_net is not None:
        if foreign_net >= 20_000:
            score += 1
        elif foreign_net <= -20_000:
            score -= 1

    if score >= 2:
        return "Risk-on"
    if score <= -2:
        return "Risk-off"
    return "Neutral"


def collect_twse_market_breadth(curr_date: str) -> dict[str, Any]:
    # Taiwan equity market data for TWSE breadth and headline index signals.
    trade_date = _parse_trade_date(curr_date)
    result: dict[str, Any] = {
        "date": trade_date.strftime("%Y-%m-%d"),
        "source": "",
        "indices": {
            "weighted": None,
            "electronics": None,
            "financials": None,
        },
        "breadth": {
            "up": None,
            "down": None,
            "flat": None,
            "untraded": None,
            "no_quote": None,
            "advance_decline_ratio": None,
        },
        "errors": [],
    }

    params = {"response": "json", "date": trade_date.strftime("%Y%m%d"), "type": "ALL"}
    # Keep both language endpoints: both are valid, and either can serve as a fallback.
    urls = [
        "https://www.twse.com.tw/rwd/en/afterTrading/MI_INDEX",
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
    ]

    payload: Optional[dict[str, Any]] = None
    for url in urls:
        try:
            response = _session_get(url, params=params, timeout=30)
            response.raise_for_status()
            candidate_payload = response.json()
            if candidate_payload.get("tables"):
                payload = candidate_payload
                result["source"] = url
                break
            result["errors"].append(f"TWSE response from {url} did not include tables.")
        except Exception as exc:  # pragma: no cover - network path
            result["errors"].append(f"TWSE request failed ({url}): {_summarize_exception(exc)}")

    if not payload:
        return result

    payload_date = payload.get("date")
    if payload_date:
        result["date"] = str(payload_date)

    tables = payload.get("tables", [])
    price_table = None
    breadth_table = None
    for table in tables:
        title = str(table.get("title", ""))
        if any(
            token in title
            for token in [
                "TWSE Indices",
                "Price Index",
                "TAIEX",
                "價格指數",
            ]
        ):
            price_table = table
        if any(
            token in title
            for token in [
                "Net Change of Price",
                "Number of Listed Securities",
                "Advance",
                "Decline",
                "漲跌證券數合計",
                "大盤統計資訊",
            ]
        ):
            breadth_table = table

    if price_table:
        rows = price_table.get("data", [])
        weighted = _find_row_by_aliases(rows, ["TAIEX", "發行量加權股價指數"])
        electronics = _find_row_by_aliases(rows, ["Electronics", "電子類指數"])
        finance = _find_row_by_aliases(rows, ["Finance and Insurance", "金融保險類指數"])

        result["indices"]["weighted"] = {
            "index": _safe_row_value(weighted, 1),
            "change": _safe_row_value(weighted, 2),
            "direction": _safe_row_value(weighted, 3),
            "change_pct": _safe_row_value(weighted, 4),
        }
        result["indices"]["electronics"] = {
            "index": _safe_row_value(electronics, 1),
            "change": _safe_row_value(electronics, 2),
            "direction": _safe_row_value(electronics, 3),
            "change_pct": _safe_row_value(electronics, 4),
        }
        result["indices"]["financials"] = {
            "index": _safe_row_value(finance, 1),
            "change": _safe_row_value(finance, 2),
            "direction": _safe_row_value(finance, 3),
            "change_pct": _safe_row_value(finance, 4),
        }
    else:
        result["errors"].append("TWSE price table not found in response payload.")

    if breadth_table:
        rows = breadth_table.get("data", [])
        breadth_map = {str(row[0]).strip(): row[1:] for row in rows if row}

        def _first_match(aliases: list[str]) -> Any:
            for key, values in breadth_map.items():
                norm_key = _normalize_text(key)
                for alias in aliases:
                    if _normalize_text(alias) in norm_key:
                        return values[0] if values else None
            return None

        up = _to_int(_first_match(["Up", "Advance", "Advancing", "上漲"]))
        down = _to_int(_first_match(["Down", "Decline", "Declining", "下跌"]))
        flat = _to_int(_first_match(["Unchanged", "持平"]))
        untraded = _to_int(_first_match(["Unmatched", "Untraded", "未成交"]))
        no_quote = _to_int(_first_match(["N/A", "No Quote", "無比價"]))
        ratio = (up / max(down, 1)) if up is not None and down is not None else None

        result["breadth"]["up"] = up
        result["breadth"]["down"] = down
        result["breadth"]["flat"] = flat
        result["breadth"]["untraded"] = untraded
        result["breadth"]["no_quote"] = no_quote
        result["breadth"]["advance_decline_ratio"] = ratio
    else:
        result["errors"].append("TWSE breadth table not found in response payload.")

    return result


def render_twse_market_breadth(data: dict[str, Any]) -> str:
    output = [f"# TWSE Market Breadth ({data.get('date', 'N/A')})"]
    indices = data.get("indices", {})

    weighted = indices.get("weighted")
    electronics = indices.get("electronics")
    financials = indices.get("financials")

    if weighted:
        output.append(
            "- Weighted index: "
            f"{weighted.get('index', 'N/A')} "
            f"({weighted.get('change', 'N/A')} {weighted.get('direction', '')}, {weighted.get('change_pct', 'N/A')}%)"
        )
    if electronics:
        output.append(
            "- Electronics: "
            f"{electronics.get('index', 'N/A')} "
            f"({electronics.get('change', 'N/A')} {electronics.get('direction', '')}, {electronics.get('change_pct', 'N/A')}%)"
        )
    if financials:
        output.append(
            "- Financials: "
            f"{financials.get('index', 'N/A')} "
            f"({financials.get('change', 'N/A')} {financials.get('direction', '')}, {financials.get('change_pct', 'N/A')}%)"
        )

    breadth = data.get("breadth", {})
    output.append("- Breadth counts:")
    output.append(f"  - Up: {_fmt_int(breadth.get('up'))}")
    output.append(f"  - Down: {_fmt_int(breadth.get('down'))}")
    output.append(f"  - Flat: {_fmt_int(breadth.get('flat'))}")
    output.append(f"  - Untraded: {_fmt_int(breadth.get('untraded'))}")
    output.append(f"  - No quote: {_fmt_int(breadth.get('no_quote'))}")
    output.append(
        "  - Advance/decline ratio: "
        f"{_fmt_float(breadth.get('advance_decline_ratio'), 2) if breadth.get('advance_decline_ratio') is not None else 'N/A'}"
    )

    if data.get("source"):
        output.append(f"- Source: {data['source']}")
    if data.get("errors"):
        output.append("- Notes:")
        for err in data["errors"]:
            output.append(f"  - {err}")

    return "\n".join(output)


def _find_identity_row(df: pd.DataFrame, aliases: list[str]) -> Optional[pd.Series]:
    normalized_aliases = [_normalize_text(alias) for alias in aliases]
    for _, row in df.iterrows():
        cell = _normalize_text(row.iloc[0]) if len(row) > 0 else ""
        for alias in normalized_aliases:
            if alias and (cell == alias or alias in cell):
                return row
    return None


def _find_contract_row(df: pd.DataFrame, product_aliases: list[str], identity_aliases: list[str]) -> Optional[pd.Series]:
    product_aliases_norm = [_normalize_text(v) for v in product_aliases]
    identity_aliases_norm = [_normalize_text(v) for v in identity_aliases]

    for _, row in df.iterrows():
        row_cells = [_normalize_text(v) for v in row.tolist()]
        has_product = any(
            alias and any(alias in cell for cell in row_cells) for alias in product_aliases_norm
        )
        has_identity = any(
            alias and any(alias in cell for cell in row_cells) for alias in identity_aliases_norm
        )
        if has_product and has_identity:
            return row
    return None


def _get_series_value(row: pd.Series, keyword_groups: list[list[str]], fallback_index: Optional[int] = None) -> Any:
    for column, value in row.items():
        if isinstance(column, tuple):
            column_name = " ".join(str(part) for part in column if part is not None)
        else:
            column_name = str(column)
        norm_col = _normalize_text(column_name)
        for keywords in keyword_groups:
            normalized_keywords = [_normalize_text(k) for k in keywords]
            if all(keyword in norm_col for keyword in normalized_keywords):
                return value

    if fallback_index is not None and fallback_index < len(row):
        return row.iloc[fallback_index]
    return None


def collect_taifex_market_regime(curr_date: str) -> dict[str, Any]:
    # Taiwan derivatives market data for TAIFEX futures / options positioning.
    trade_date = _parse_trade_date(curr_date)
    label = trade_date.strftime("%Y-%m-%d")
    result: dict[str, Any] = {
        "date": label,
        "sources": [],
        "summary": {
            "trading": {"dealer": None, "investment_trust": None, "foreign": None},
            "open_interest": {"dealer": None, "investment_trust": None, "foreign": None},
        },
        "futures": {},
        "options": {},
        "errors": [],
    }

    # English and Chinese URLs both work; keep both for stability.
    total_urls = [
        "https://www.taifex.com.tw/eng/3/totalTableDate",
        "https://www.taifex.com.tw/cht/3/totalTableDate",
    ]
    futures_urls = [
        "https://www.taifex.com.tw/eng/3/futContractsDate",
        "https://www.taifex.com.tw/cht/3/futContractsDate",
    ]
    options_urls = [
        "https://www.taifex.com.tw/eng/3/optContractsDate",
        "https://www.taifex.com.tw/cht/3/optContractsDate",
    ]

    try:
        total_df, total_source = _read_taifex_table_with_fallback(total_urls, curr_date, 0)
        result["sources"].append(total_source)
    except Exception as exc:
        total_df = pd.DataFrame()
        result["errors"].append(f"TAIFEX total trading table unavailable: {exc}")

    try:
        total_oi_df, total_oi_source = _read_taifex_table_with_fallback(total_urls, curr_date, 1)
        result["sources"].append(total_oi_source)
    except Exception as exc:
        total_oi_df = pd.DataFrame()
        result["errors"].append(f"TAIFEX total OI table unavailable: {exc}")

    try:
        futures_df, futures_source = _read_taifex_table_with_fallback(futures_urls, curr_date, 0)
        result["sources"].append(futures_source)
    except Exception as exc:
        futures_df = pd.DataFrame()
        result["errors"].append(f"TAIFEX futures table unavailable: {exc}")

    try:
        options_df, options_source = _read_taifex_table_with_fallback(options_urls, curr_date, 0)
        result["sources"].append(options_source)
    except Exception as exc:
        options_df = pd.DataFrame()
        result["errors"].append(f"TAIFEX options table unavailable: {exc}")

    def _summary_entry(row: Optional[pd.Series]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        return {
            "long": _to_int(_get_series_value(row, [["long"]], fallback_index=1)),
            "short": _to_int(_get_series_value(row, [["short"]], fallback_index=3)),
            "net": _to_int(_get_series_value(row, [["net"]], fallback_index=5)),
            "value_net": _to_int(_get_series_value(row, [["value", "net"], ["amount", "net"]], fallback_index=6)),
        }

    if not total_df.empty:
        result["summary"]["trading"]["dealer"] = _summary_entry(_find_identity_row(total_df, ["Dealer", "自營商"]))
        result["summary"]["trading"]["investment_trust"] = _summary_entry(
            _find_identity_row(total_df, ["Investment Trust", "投信"])
        )
        result["summary"]["trading"]["foreign"] = _summary_entry(_find_identity_row(total_df, ["Foreign", "外資"]))

    if not total_oi_df.empty:
        result["summary"]["open_interest"]["dealer"] = _summary_entry(
            _find_identity_row(total_oi_df, ["Dealer", "自營商"])
        )
        result["summary"]["open_interest"]["investment_trust"] = _summary_entry(
            _find_identity_row(total_oi_df, ["Investment Trust", "投信"])
        )
        result["summary"]["open_interest"]["foreign"] = _summary_entry(
            _find_identity_row(total_oi_df, ["Foreign", "外資"])
        )

    futures_focus = {
        "TAIEX Futures": ["TAIEX Futures", "臺股期貨"],
        "Mini TAIEX Futures": ["Mini TAIEX Futures", "小型臺指期貨"],
        "Micro TAIEX Futures": ["Micro TAIEX Futures", "微型臺指期貨"],
        "Electronics Futures": ["Electronics Futures", "電子期貨"],
        "Finance Futures": ["Finance Futures", "金融期貨"],
        "Stock Futures": ["Stock Futures", "股票期貨"],
    }
    options_focus = {
        "TAIEX Options": ["TAIEX Options", "臺指選擇權"],
        "Electronics Options": ["Electronics Options", "電子選擇權"],
        "Finance Options": ["Finance Options", "金融選擇權"],
        "Stock Options": ["Stock Options", "股票選擇權"],
        "ETF Options": ["ETF Options", "ETF選擇權"],
    }
    identities = {
        "dealer": ["Dealer", "自營商"],
        "investment_trust": ["Investment Trust", "投信"],
        "foreign": ["Foreign", "外資"],
    }

    def _contract_metrics(row: Optional[pd.Series]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        return {
            "trade_net": _to_int(
                _get_series_value(
                    row,
                    [["tradingvolume", "net", "contracts"], ["交易口數與契約金額", "多空淨額", "口數"]],
                )
            ),
            "oi_long": _to_int(
                _get_series_value(
                    row,
                    [["openinterest", "long", "contracts"], ["未平倉餘額", "多方", "口數"]],
                )
            ),
            "oi_short": _to_int(
                _get_series_value(
                    row,
                    [["openinterest", "short", "contracts"], ["未平倉餘額", "空方", "口數"]],
                )
            ),
            "oi_net": _to_int(
                _get_series_value(
                    row,
                    [["openinterest", "net", "contracts"], ["未平倉餘額", "多空淨額", "口數"]],
                )
            ),
        }

    if not futures_df.empty:
        for contract_name, aliases in futures_focus.items():
            block: dict[str, dict[str, Any]] = {}
            for identity_name, identity_aliases in identities.items():
                row = _find_contract_row(futures_df, aliases, identity_aliases)
                metrics = _contract_metrics(row)
                if metrics:
                    block[identity_name] = metrics
            if block:
                result["futures"][contract_name] = block

    if not options_df.empty:
        for contract_name, aliases in options_focus.items():
            block = {}
            for identity_name, identity_aliases in identities.items():
                row = _find_contract_row(options_df, aliases, identity_aliases)
                metrics = _contract_metrics(row)
                if metrics:
                    block[identity_name] = metrics
            if block:
                result["options"][contract_name] = block

    # Keep source list compact and deterministic.
    result["sources"] = sorted(set(result["sources"]))
    return result


def render_taifex_market_regime(data: dict[str, Any]) -> str:
    out = [f"# TAIFEX Market Regime ({data.get('date', 'N/A')})"]
    summary = data.get("summary", {})

    def _line(entry: Optional[dict[str, Any]], prefix: str) -> str:
        if not entry:
            return f"- {prefix}: N/A"
        return (
            f"- {prefix}: long {_fmt_int(entry.get('long'))}, short {_fmt_int(entry.get('short'))}, "
            f"net {_fmt_int(entry.get('net'))}; value net {_fmt_int(entry.get('value_net'))}"
        )

    out.append("## Total table")
    trading = summary.get("trading", {})
    open_interest = summary.get("open_interest", {})
    out.append(_line(trading.get("dealer"), "Trading / Dealer"))
    out.append(_line(trading.get("investment_trust"), "Trading / Investment trust"))
    out.append(_line(trading.get("foreign"), "Trading / Foreign"))
    out.append(_line(open_interest.get("dealer"), "Open interest / Dealer"))
    out.append(_line(open_interest.get("investment_trust"), "Open interest / Investment trust"))
    out.append(_line(open_interest.get("foreign"), "Open interest / Foreign"))

    out.append("## Futures contract focus")
    for contract_name, block in data.get("futures", {}).items():
        out.append(f"- {contract_name}")
        for identity, metrics in block.items():
            identity_title = identity.replace("_", " ").title()
            out.append(
                f"  - {identity_title}: trade net {_fmt_int(metrics.get('trade_net'))}, "
                f"OI long {_fmt_int(metrics.get('oi_long'))}, "
                f"OI short {_fmt_int(metrics.get('oi_short'))}, "
                f"OI net {_fmt_int(metrics.get('oi_net'))}"
            )

    out.append("## Options contract focus")
    for contract_name, block in data.get("options", {}).items():
        out.append(f"- {contract_name}")
        for identity, metrics in block.items():
            identity_title = identity.replace("_", " ").title()
            out.append(
                f"  - {identity_title}: trade net {_fmt_int(metrics.get('trade_net'))}, "
                f"OI long {_fmt_int(metrics.get('oi_long'))}, "
                f"OI short {_fmt_int(metrics.get('oi_short'))}, "
                f"OI net {_fmt_int(metrics.get('oi_net'))}"
            )

    sources = data.get("sources", [])
    if sources:
        out.append(f"- Sources: {', '.join(sources)}")
    if data.get("errors"):
        out.append("- Notes:")
        for err in data["errors"]:
            out.append(f"  - {err}")

    return "\n".join(out)


def collect_vix_fx_snapshot(curr_date: str) -> dict[str, Any]:
    # Macro context used by Taiwan stock analysis: VIX and USD/TWD.
    trade_date = _parse_trade_date(curr_date)
    result: dict[str, Any] = {
        "date": trade_date.strftime("%Y-%m-%d"),
        "vix": None,
        "usd_twd": None,
        "source": "yfinance",
        "errors": [],
    }

    vix = _series_from_yfinance("^VIX", curr_date)
    fx = _series_from_yfinance("USDTWD=X", curr_date)
    if fx is None:
        fx = _series_from_yfinance("TWD=X", curr_date)

    if vix is None:
        result["errors"].append("VIX series unavailable from yfinance.")
    else:
        result["vix"] = {
            "close": _to_float(vix.get("Close")),
            "open": _to_float(vix.get("Open")),
            "high": _to_float(vix.get("High")),
            "low": _to_float(vix.get("Low")),
        }

    if fx is None:
        result["errors"].append("USD/TWD series unavailable from yfinance (USDTWD=X, TWD=X).")
    else:
        result["usd_twd"] = {
            "close": _to_float(fx.get("Close")),
            "open": _to_float(fx.get("Open")),
            "high": _to_float(fx.get("High")),
            "low": _to_float(fx.get("Low")),
        }

    return result


def collect_twse_institutional_spot_flow(curr_date: str) -> dict[str, Any]:
    trade_date = _parse_trade_date(curr_date)
    result: dict[str, Any] = {
        "date": trade_date.strftime("%Y-%m-%d"),
        "source": "",
        "spot_net": {
            "foreign": None,
            "investment_trust": None,
            "dealer": None,
            "total": None,
        },
        "errors": [],
    }

    urls = [
        "https://www.twse.com.tw/rwd/en/fund/BFI82U",
        "https://www.twse.com.tw/rwd/zh/fund/BFI82U",
    ]
    params = {"response": "json", "dayDate": trade_date.strftime("%Y%m%d"), "type": "day"}

    payload: Optional[dict[str, Any]] = None
    for url in urls:
        try:
            response = _session_get(url, params=params, timeout=30)
            response.raise_for_status()
            candidate = response.json()
            if candidate.get("data"):
                payload = candidate
                result["source"] = url
                break
            result["errors"].append(f"TWSE institutional flow response from {url} did not include data.")
        except Exception as exc:
            result["errors"].append(
                f"TWSE institutional flow request failed ({url}): {_summarize_exception(exc)}"
            )

    if not payload:
        return result

    payload_date = payload.get("date")
    if payload_date:
        result["date"] = str(payload_date)

    rows = payload.get("data", [])
    flow_map: dict[str, Any] = {}
    for row in rows:
        if len(row) < 4:
            continue
        flow_map[_normalize_text(row[0])] = row[3]

    def _match_diff(aliases: list[str]) -> Optional[int]:
        aliases_norm = [_normalize_text(alias) for alias in aliases]
        for key, value in flow_map.items():
            for alias in aliases_norm:
                if alias in key:
                    return _to_int(value)
        return None

    dealer_prop = _match_diff(["dealers(proprietary)", "自營商(自行買賣)"])
    dealer_hedge = _match_diff(["dealers(hedge)", "自營商(避險)"])
    dealer_total = None
    if dealer_prop is not None or dealer_hedge is not None:
        dealer_total = (dealer_prop or 0) + (dealer_hedge or 0)

    result["spot_net"]["foreign"] = _match_diff(
        ["foreigninvestorsincludemainlandareainvestors", "外資及陸資(不含外資自營商)"]
    )
    result["spot_net"]["investment_trust"] = _match_diff(
        ["securitiesinvestmenttrustcompanies", "投信"]
    )
    result["spot_net"]["dealer"] = dealer_total
    result["spot_net"]["total"] = _match_diff(["total", "合計"])

    return result


def collect_margin_maintenance_ratio(curr_date: str) -> dict[str, Any]:
    trade_date = _parse_trade_date(curr_date)
    result: dict[str, Any] = {
        "date": trade_date.strftime("%Y-%m-%d"),
        "source": "",
        "margin_maintenance_ratio": None,
        "errors": [],
    }

    urls = [
        "https://www.twse.com.tw/rwd/en/marginTrading/MI_MARGN",
        "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
    ]
    params = {"response": "json", "date": trade_date.strftime("%Y%m%d"), "selectType": "MS"}

    payload: Optional[dict[str, Any]] = None
    for url in urls:
        try:
            response = _session_get(url, params=params, timeout=30)
            response.raise_for_status()
            candidate = response.json()
            if candidate.get("tables"):
                payload = candidate
                result["source"] = url
                break
            result["errors"].append(f"Margin response from {url} did not include tables.")
        except Exception as exc:
            result["errors"].append(f"Margin request failed ({url}): {_summarize_exception(exc)}")

    if not payload:
        return result

    payload_date = payload.get("date")
    if payload_date:
        result["date"] = str(payload_date)

    # TWSE MI_MARGN does not consistently publish market-level maintenance ratio in JSON.
    for table in payload.get("tables", []):
        for row in table.get("data", []) or []:
            if not row:
                continue
            label = _normalize_text(row[0])
            if "維持率" in str(row[0]) or "maintenance" in label:
                for value in row[1:]:
                    parsed = _to_float(value)
                    if parsed is not None:
                        result["margin_maintenance_ratio"] = parsed
                        return result

    result["errors"].append("Margin maintenance ratio is not present in TWSE MI_MARGN JSON for this date.")
    return result


def collect_analyst_market_regime_schema(curr_date: str) -> dict[str, Any]:
    taifex_data = collect_taifex_market_regime(curr_date)
    spot_data = collect_twse_institutional_spot_flow(curr_date)
    macro_data = collect_vix_fx_snapshot(curr_date)

    taiex_foreign_oi = _to_int(
        (((taifex_data.get("futures") or {}).get("TAIEX Futures") or {}).get("foreign") or {}).get("oi_net")
    )
    if taiex_foreign_oi is None:
        taiex_foreign_oi = _to_int(
            (((taifex_data.get("summary") or {}).get("open_interest") or {}).get("foreign") or {}).get("net")
        )

    schema: dict[str, Any] = {
        "date": curr_date,
        "signals": {
            "taiex_futures_foreign_oi_net": taiex_foreign_oi,
            "spot_net_buy_sell": {
                "foreign": _to_int((spot_data.get("spot_net") or {}).get("foreign")),
                "investment_trust": _to_int((spot_data.get("spot_net") or {}).get("investment_trust")),
                "dealer": _to_int((spot_data.get("spot_net") or {}).get("dealer")),
            },
            "vix_close": _to_float(((macro_data.get("vix") or {}).get("close"))),
            "usd_twd_close": _to_float(((macro_data.get("usd_twd") or {}).get("close"))),
        },
        "sources": {
            "taifex": taifex_data.get("sources", []),
            "twse_spot": spot_data.get("source", ""),
            "macro": macro_data.get("source", ""),
        },
        "errors": [],
    }

    schema["errors"].extend(taifex_data.get("errors", []))
    schema["errors"].extend(spot_data.get("errors", []))
    schema["errors"].extend(macro_data.get("errors", []))
    return schema


def render_vix_fx_snapshot(data: dict[str, Any]) -> str:
    out = [f"# VIX / FX Snapshot ({data.get('date', 'N/A')})"]
    vix = data.get("vix")
    fx = data.get("usd_twd")

    if vix is not None:
        out.append(
            f"- VIX: close {_fmt_float(vix.get('close'))}, open {_fmt_float(vix.get('open'))}, "
            f"high {_fmt_float(vix.get('high'))}, low {_fmt_float(vix.get('low'))}"
        )
    else:
        out.append("- VIX: unavailable")

    if fx is not None:
        out.append(
            f"- USD/TWD: close {_fmt_float(fx.get('close'), 4)}, open {_fmt_float(fx.get('open'), 4)}, "
            f"high {_fmt_float(fx.get('high'), 4)}, low {_fmt_float(fx.get('low'), 4)}"
        )
    else:
        out.append("- USD/TWD: unavailable")

    if data.get("source"):
        out.append(f"- Source: {data['source']}")
    if data.get("errors"):
        out.append("- Notes:")
        for err in data["errors"]:
            out.append(f"  - {err}")

    return "\n".join(out)


@tool
def get_twse_market_breadth(curr_date: str) -> str:
    """Get TWSE broad-market breadth and headline price index data for a date."""
    data = collect_twse_market_breadth(curr_date)
    return render_twse_market_breadth(data)


@tool
def get_taifex_market_regime(curr_date: str) -> str:
    """Get TAIFEX futures and options regime statistics for a date."""
    data = collect_taifex_market_regime(curr_date)
    return render_taifex_market_regime(data)


@tool
def get_vix_fx_snapshot(curr_date: str) -> str:
    """Get VIX and USD/TWD market snapshots from yfinance."""
    data = collect_vix_fx_snapshot(curr_date)
    return render_vix_fx_snapshot(data)


@tool
def get_market_regime_summary(curr_date: str) -> str:
    # Single entry point for Taiwan stock market regime analysis.
    """Build a combined market regime summary from TWSE, TAIFEX, VIX, and FX data."""
    history_rows: list[dict[str, Any]] = []
    base_date = _parse_trade_date(curr_date)
    for offset in range(0, 7):
        day = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        day_schema = collect_analyst_market_regime_schema(day)
        day_signals = day_schema.get("signals", {})
        history_rows.append(
            {
                "date": day,
                "foreign_taiex_oi_net": _to_int(day_signals.get("taiex_futures_foreign_oi_net")),
                "spot_net_buy_sell": {
                    "foreign": _to_int((day_signals.get("spot_net_buy_sell") or {}).get("foreign")),
                    "investment_trust": _to_int(
                        (day_signals.get("spot_net_buy_sell") or {}).get("investment_trust")
                    ),
                    "dealer": _to_int((day_signals.get("spot_net_buy_sell") or {}).get("dealer")),
                },
                "vix_close": _to_float(day_signals.get("vix_close")),
                "usd_twd_close": _to_float(day_signals.get("usd_twd_close")),
            }
        )

    payload = {
        "as_of": curr_date,
        "window_days": 7,
        "signal_trend": history_rows,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)