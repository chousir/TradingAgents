from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

        tools = [
            get_market_regime_summary,
        ]

        system_message = (
            """You are a macro / market-regime analyst.

Your job is to summarize the broad market tone before the stock-specific analysts speak. Use official market-wide signals and keep the output factual, compact, and actionable.

Priority signals:
- TWSE breadth and headline index behavior
- TAIFEX futures / options regime and foreign open-interest pressure
- VIX and USD/TWD context

Use the tools to gather the market regime. Then write a concise report with:
- One-line regime conclusion.
- Key evidence bullets.
- A short trader-facing bias statement that says whether the environment is supportive, mixed, or defensive for the named ticker.
- If the data conflict, say so explicitly.

Do not repeat company fundamentals or detailed price targets. Do not make the final trade decision."""
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
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "macro_report": report,
        }

    return macro_regime_analyst_node