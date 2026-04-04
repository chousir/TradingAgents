from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime, timedelta
import csv
import io
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def _extract_latest_close(stock_data_csv: str):
    lines = [line for line in stock_data_csv.splitlines() if line and not line.startswith("#")]
    if len(lines) < 2:
        return None, None

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        return None, None

    last_row = rows[-1]
    latest_close = last_row.get("Close") or last_row.get("Adj Close")
    latest_date = last_row.get("Date")

    try:
        latest_close = float(latest_close) if latest_close not in (None, "") else None
    except (TypeError, ValueError):
        latest_close = None

    return latest_close, latest_date


def _summarize_price_structure(stock_data_csv: str):
    lines = [line for line in stock_data_csv.splitlines() if line and not line.startswith("#")]
    if len(lines) < 2:
        return {}

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        return {}

    recent_rows = rows[-20:] if len(rows) >= 20 else rows

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    closes = [_to_float(row.get("Close") or row.get("Adj Close")) for row in recent_rows]
    highs = [_to_float(row.get("High")) for row in recent_rows]
    lows = [_to_float(row.get("Low")) for row in recent_rows]

    closes = [value for value in closes if value is not None]
    highs = [value for value in highs if value is not None]
    lows = [value for value in lows if value is not None]

    if not closes or not highs or not lows:
        return {}

    latest_close = closes[-1]
    recent_high = max(highs)
    recent_low = min(lows)
    range_width = max(recent_high - recent_low, 0.0)
    support_zone_low = recent_low
    support_zone_high = recent_low + range_width * 0.25 if range_width else recent_low
    resistance_zone_low = recent_high - range_width * 0.25 if range_width else recent_high
    resistance_zone_high = recent_high

    distance_to_support = latest_close - support_zone_high
    distance_to_resistance = resistance_zone_low - latest_close

    return {
        "recent_high": round(recent_high, 4),
        "recent_low": round(recent_low, 4),
        "support_zone": (round(support_zone_low, 4), round(support_zone_high, 4)),
        "resistance_zone": (round(resistance_zone_low, 4), round(resistance_zone_high, 4)),
        "range_width": round(range_width, 4),
        "distance_to_support": round(distance_to_support, 4),
        "distance_to_resistance": round(distance_to_resistance, 4),
    }


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        current_date_dt = datetime.strptime(current_date, "%Y-%m-%d")
        price_start_date = (current_date_dt - timedelta(days=45)).strftime("%Y-%m-%d")
        price_data = get_stock_data.invoke(
            {
                "symbol": ticker,
                "start_date": price_start_date,
                "end_date": current_date,
            }
        )
        latest_close_price, latest_close_date = _extract_latest_close(price_data)
        price_structure = _summarize_price_structure(price_data)
        latest_close_context = (
            f"Latest available close price for {ticker}: {latest_close_price} on {latest_close_date}."
            if latest_close_price is not None and latest_close_date
            else f"Latest available close price for {ticker}: unavailable."
        )
        price_structure_context = (
            "Price structure summary: "
            f"recent_high={price_structure.get('recent_high')}, "
            f"recent_low={price_structure.get('recent_low')}, "
            f"support_zone={price_structure.get('support_zone')}, "
            f"resistance_zone={price_structure.get('resistance_zone')}, "
            f"range_width={price_structure.get('range_width')}, "
            f"distance_to_support={price_structure.get('distance_to_support')}, "
            f"distance_to_resistance={price_structure.get('distance_to_resistance')}."
            if price_structure
            else "Price structure summary unavailable."
        )

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a market structure analyst and scenario planner.

Your job is not to issue the final trade, but to produce a compact price map and 2-3 executable scenarios for a trader.

Workflow:
1) Call `get_stock_data` first.
2) Then call `get_indicators` using exact indicator names only.
3) Use the price data, indicators, and price structure summary to build scenarios.

Allowed indicators:
close_10_ema, close_50_sma, close_200_sma, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma.

Prefer grouped indicator calls when possible.

Output must include:
- A one-line market regime conclusion.
- A price map with latest close, support zone, resistance zone, and volatility context.
- 2 to 3 scenarios only:
    - Breakout / momentum scenario
    - Pullback / mean-reversion scenario
    - Failure / no-trade scenario
- For each scenario, provide:
    - Entry zone
    - Stop loss / invalidation
    - Take profit zone(s)
    - Tranche idea if relevant
- Finish with a short markdown table summarizing the scenarios.

Keep the output concise, numeric, and trader-ready. Do not write the final portfolio decision; leave that to the Trader and Portfolio Manager."""
        + get_language_instruction()
    )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context} {latest_close_context} {price_structure_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(latest_close_context=latest_close_context)
        prompt = prompt.partial(price_structure_context=price_structure_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
            "latest_close_price": latest_close_price,
            "latest_close_date": latest_close_date,
            "price_structure": price_structure,
        }

    return market_analyst_node
