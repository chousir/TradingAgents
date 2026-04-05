from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_market_regime_summary,
)


def create_macro_regime_analyst(llm):

    def macro_regime_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        market_regime_json = get_market_regime_summary.invoke({"curr_date": current_date})

        system_message = (
            f"""You are a macro / market-regime analyst.

Your job is to summarize the broad market tone before the stock-specific analysts speak. Use official market-wide signals and keep the output factual, compact, and actionable.

Hard data cutoff rule:
- Treat {current_date} as the strict cutoff date.
- Do not reference or infer data after {current_date}.
- If weekend/non-trading-day flags appear, explain them as expected market-closed context.

Priority signals:
- TWSE breadth and headline index behavior
- TAIFEX futures / options regime and foreign open-interest pressure
- VIX and USD/TWD context

Use the tools to gather the market regime. Then write a concise report with:
- One-line regime conclusion.
- Key evidence bullets.
- A short trader-facing bias statement that says whether the environment is supportive, mixed, or defensive for the named ticker.
- If the data conflict, say so explicitly.
- Keep the report concise and end with the exact line: END OF REPORT.

Do not repeat company fundamentals or detailed price targets. Do not make the final trade decision."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants. "
                    "Use the provided market regime JSON as the factual source of truth and do not fabricate data.\n"
                    "{system_message}",
                ),
                (
                    "human",
                    "Current date: {current_date}\n"
                    "{instrument_context}\n\n"
                    "Market regime JSON (strict data source):\n{market_regime_json}",
                ),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(market_regime_json=market_regime_json)

        result = llm.invoke(prompt.format_messages())

        report = result.content

        return {
            "messages": [result],
            "macro_report": report,
        }

    return macro_regime_analyst_node