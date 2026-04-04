import functools
import time
import json

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state.get("investment_plan", "No plan provided")
        latest_close_price = state.get("latest_close_price")
        latest_close_date = state.get("latest_close_date")
        price_structure = state.get("price_structure", {})
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is the full decision context for {company_name}. {instrument_context} Use the analyst reports below as the factual basis for your price levels, risk levels, and tranche sizing. If you cannot derive an exact price, provide the best support/resistance-based estimate and state the assumption clearly.\n\nLatest Available Close Price: {latest_close_price if latest_close_price is not None else 'Unavailable'}\nLatest Available Close Date: {latest_close_date if latest_close_date else 'Unavailable'}\nPrice Structure Summary: {price_structure if price_structure else 'Unavailable'}\n\nMarket Report:\n{market_research_report}\n\nSentiment Report:\n{sentiment_report}\n\nNews Report:\n{news_report}\n\nFundamentals Report:\n{fundamentals_report}\n\nProposed Investment Plan:\n{investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on the technical and fundamental analysis provided, you MUST provide a specific recommendation to buy, sell, or hold.

Crucially, your proposed plan must include concrete actionable figures:
1. Target Entry Price / Range: specific price levels to enter the trade.
2. Take Profit (TP) Target(s): specific price points to take profit.
3. Stop Loss (SL): a strict price level to cut losses.
4. Position Sizing & Scaling Strategy (Tranches): how to scale into the position, for example 30% at market and 70% at support limit.

Use the analyst reports, the latest close, and the price structure summary as the factual basis for your numbers. If exact prices are not directly stated, infer them from the market report's support/resistance or recent price structure and clearly mark them as estimates.

End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'. Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
