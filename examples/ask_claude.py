"""Ask Claude about a video without sending it every frame.

    pip install framebudget anthropic
    python examples/ask_claude.py talk.mp4 "what changes between the slides?"

The budget below is for images only. Leave room for the question, the system
prompt and the answer on top of it.
"""

from __future__ import annotations

import sys

import anthropic

from framebudget import extract

BUDGET = 50_000
MODEL = "claude-opus-5"


def main(path: str, question: str) -> int:
    result = extract(path, budget=BUDGET, target="claude")
    print(result.report.summary(), file=sys.stderr)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16_000,
        messages=[
            {
                "role": "user",
                # Frames first, question last. The timestamps interleaved with the
                # images are what let the model answer anything about ordering or
                # duration, which is most of what people ask about video.
                "content": [*result.to_messages(), {"type": "text", "text": question}],
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
