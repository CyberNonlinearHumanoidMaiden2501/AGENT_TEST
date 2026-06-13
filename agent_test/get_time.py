import os
import time
import json
import urllib.request
from openai import OpenAI

# ---------------------------------------------------------------------------
# DeepSeek client (same skeleton as before)
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ---------------------------------------------------------------------------
# Tool definition — the model can call get_current_time to fetch real time
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time from an internet time source "
                "(WorldTimeAPI). Returns accurate UTC datetime, timezone "
                "information, and day-of-week. Call this every time you need "
                "to report the current time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "Optional IANA timezone string "
                            "(e.g. 'America/New_York', 'Asia/Shanghai'). "
                            "Defaults to 'Etc/UTC' when omitted."
                        ),
                    }
                },
                "required": [],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Tool implementation — actually fetches time from the internet
# ---------------------------------------------------------------------------
def execute_get_current_time(timezone: str = "Etc/UTC") -> str:
    """Call the WorldTimeAPI and return the JSON response as a string."""
    url = f"https://worldtimeapi.org/api/timezone/{timezone}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    # Return a compact, human-readable subset so the model can present it nicely
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# One cycle: ask the model for the time, let it call the tool, and print
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise time-reporting assistant. "
    "Every time the user asks for the current date and time you MUST call "
    "the get_current_time tool to fetch it from the internet. "
    "Never guess or rely on your training data — always use the tool. "
    "After you receive the tool result, report the date and time in a clear, "
    "human-friendly format."
)

USER_PROMPT = "Please tell me the current date and time."


def run_one_cycle() -> None:
    """Run a single check — prompt → tool-call → tool-result → final answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
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

    # If the model didn't call the tool we still handle it gracefully
    if choice.message.tool_calls is None:
        print("[warn] Model did not call the tool — using fallback response:")
        print(choice.message.content or "[no content]")
        return

    # --- Execute every tool call the model requested ---------------------------
    for tool_call in choice.message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        tz = args.get("timezone", "Etc/UTC")
        print(f"[tool] Model requested {name}(timezone={tz!r})")

        try:
            result = execute_get_current_time(tz)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})

        # Append the assistant's tool-call message and the tool result
        messages.append(choice.message)  # assistant message with tool_calls
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Second call — model now has the tool result and produces the final answer
    final_response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    answer = final_response.choices[0].message.content
    print(answer)


# ---------------------------------------------------------------------------
# Main loop — check every 30 minutes
# ---------------------------------------------------------------------------
INTERVAL_SECONDS = 30 * 60  # 30 minutes

if __name__ == "__main__":
    print(f"Starting time-checking agent (interval: {INTERVAL_SECONDS}s)")
    while True:
        try:
            run_one_cycle()
        except Exception as exc:
            print(f"[error] Cycle failed: {exc}")
        print(f"[sleep] Waiting {INTERVAL_SECONDS // 60} minutes until next check…")
        time.sleep(INTERVAL_SECONDS)
