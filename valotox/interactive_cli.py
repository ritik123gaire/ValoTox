"""Interactive terminal session: type text, print toxicity classifier output."""

from __future__ import annotations

import sys


def run_interactive(threshold: float = 0.5, text: str | None = None) -> None:
    """Run once with ``text``, or start a REPL until quit/EOF."""
    from valotox.agent.tools import classify_toxicity

    def _print_result(res) -> None:
        data = res.model_dump() if hasattr(res, "model_dump") else res
        labels = data.get("labels", {})
        print()
        print(f"  is_toxic:    {data.get('is_toxic')}")
        print(f"  severity:    {data.get('severity')}")
        print(f"  active:      {', '.join(data.get('active_labels') or []) or '(none)'}")
        print("  scores:")
        for name in sorted(labels.keys()):
            print(f"    {name:16s} {labels[name]:.3f}")
        print()

    if text is not None:
        line = text.strip()
        if not line:
            print("Empty text.", file=sys.stderr)
            sys.exit(1)
        _print_result(classify_toxicity(line, threshold=threshold))
        return

    print(
        "ValoTox — type a comment to classify. "
        "Empty line, quit, or Ctrl+Z then Enter (Windows) / Ctrl+D (Unix) to exit."
    )
    print(f"Classification threshold: {threshold}")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        _print_result(classify_toxicity(line, threshold=threshold))
