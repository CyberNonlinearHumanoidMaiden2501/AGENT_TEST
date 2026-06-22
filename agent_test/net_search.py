import os
import json
import urllib.request
import urllib.parse
from openai import OpenAI

# ---------------------------------------------------------------------------
# DeepSeek client (same skeleton as get_time.py)
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ---------------------------------------------------------------------------
# Tool definition — the model can call search_internet to search the web
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": (
                "Search the internet using DuckDuckGo for current, factual "
                "information. Returns an abstract/summary answer, related "
                "topic snippets, and URLs. "
                "Call this EVERY time the user asks a question that requires "
                "up-to-date or factual information you cannot answer from "
                "training data alone. Never guess — always search when you "
                "are uncertain about facts, dates, events, or real-time data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query string. Be specific and concise "
                            "(e.g. 'Python 3.13 release date', "
                            "'weather Tokyo today', "
                            "'capital of Bhutan')."
                        ),
                    },
                    "num_results": {
                        "type": "integer",
                        "description": (
                            "Optional. Maximum number of related topic results "
                            "to return. Defaults to 5. Use higher values for "
                            "broad questions, lower for focused lookups."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Tool implementation — actually searches DuckDuckGo
# ---------------------------------------------------------------------------
def execute_search_internet(query: str, num_results: int = 5) -> str:
    """Call the DuckDuckGo Instant Answer API and return JSON results."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.duckduckgo.com/"
        f"?q={encoded}&format=json&no_html=1&skip_disambig=1"
    )

    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    # Build a compact, LLM-friendly result (not the raw API dump)
    result = {
        "query": query,
        "abstract": data.get("Abstract", "") or "",
        "abstract_text": data.get("AbstractText", "") or "",
        "abstract_url": data.get("AbstractURL", "") or "",
        "answer": data.get("Answer", "") or "",  # instant answer (e.g. calculator)
        "answer_type": data.get("AnswerType", "") or "",
        "related_topics": [],
    }

    for topic in data.get("RelatedTopics", [])[:num_results]:
        if "Text" in topic and "FirstURL" in topic:
            result["related_topics"].append(
                {
                    "text": topic["Text"],
                    "url": topic["FirstURL"],
                }
            )

    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# System prompt — instructs the model HOW to use the tool
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful internet-search assistant. "
    "You MUST call the search_internet tool for any question that requires "
    "factual, current, or real-world information that you cannot answer "
    "with certainty from your training data. "
    "Never guess or fabricate facts — always search first. "
    "After you receive the search results, synthesize a clear, concise "
    "answer for the user. When the results include URLs, cite them so the "
    "user can verify the information."
)

# ---------------------------------------------------------------------------
# One cycle — prompt → tool-call → tool-result → final answer
# ---------------------------------------------------------------------------
def run_one_cycle(user_question: str) -> str | None:
    """Run a single search cycle and return the final answer (or None)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    # First call — model should request the tool
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=TOOLS,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    choice = response.choices[0]

    # Graceful fallback if the model didn't call the tool
    if choice.message.tool_calls is None:
        print("[warn] Model did not call search_internet — fallback:")
        print(choice.message.content or "[no content]")
        return choice.message.content

    # Append the assistant message (with tool_calls) ONCE before the loop
    messages.append(choice.message)

    # --- Execute every tool call the model requested -----------------------
    for tool_call in choice.message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        query = args.get("query", "")
        num = args.get("num_results", 5)
        print(f"[tool] Model requested {name}(query={query!r}, num_results={num})")

        try:
            result = execute_search_internet(query, num)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Second call — model now has the search results, produces final answer
    final_response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    answer = final_response.choices[0].message.content
    print(answer)
    return answer


# ---------------------------------------------------------------------------
# Main — interactive REPL loop (press Enter with empty input to quit)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Internet Search Agent (DuckDuckGo)")
    print("Enter your question below. Empty line to quit.\n")

    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            print("Goodbye.")
            break

        try:
            run_one_cycle(question)
        except Exception as exc:
            print(f"[error] Cycle failed: {exc}")

        print()  # blank line between Q&A pairs
