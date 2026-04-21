"""Local Gradio UI for ValoTox."""

from __future__ import annotations

from html import escape

import gradio as gr

from valotox.agent.tools import classify_toxicity


def _format_scores(scores: dict[str, float]) -> str:
    rows = ["<table><thead><tr><th>Label</th><th>Score</th></tr></thead><tbody>"]
    for label, value in scores.items():
        rows.append(f"<tr><td>{escape(label)}</td><td>{value:.4f}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def analyze_text(text: str, threshold: float = 0.5) -> tuple[str, str, str, str]:
    """Return formatted HTML for the live UI."""
    text = (text or "").strip()
    if not text:
        return (
            "<div class='empty-state'>Enter a message to classify.</div>",
            "<div class='empty-state'>No result yet.</div>",
            "<div class='empty-state'>No labels.</div>",
            "",
        )

    result = classify_toxicity(text, threshold=threshold)
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    labels = data.get("labels", {})
    active = data.get("active_labels") or []

    header_class = "danger" if data.get("is_toxic") else "safe"
    summary = f"""
    <div class='summary {header_class}'>
      <div class='kicker'>{'TOXIC' if data.get('is_toxic') else 'CLEAN'}</div>
      <div class='severity'>Severity: {escape(str(data.get('severity', 'none')))}</div>
      <div class='active'>Active labels: {escape(', '.join(active) if active else '(none)')}</div>
    </div>
    """

    score_html = _format_scores(labels)
    raw_html = f"<pre>{escape(str(data))}</pre>"
    note = (
        "<div class='note'>Heuristic fallback is used only if no loadable checkpoint exists.</div>"
        if getattr(result, "labels", None)
        else ""
    )
    return summary, score_html, raw_html, note


def build_app() -> gr.Blocks:
    """Create the Gradio Blocks app."""
    css = """
    .gradio-container { background: linear-gradient(135deg, #09111f 0%, #101827 45%, #17112a 100%); }
    .title-wrap { margin-bottom: 1rem; }
    .title-wrap h1 { font-size: 2.3rem; margin: 0; color: #f5f7ff; letter-spacing: -0.04em; }
    .title-wrap p { color: #b7c2e0; margin-top: 0.35rem; }
    .summary { border-radius: 16px; padding: 18px 20px; color: #f5f7ff; border: 1px solid rgba(255,255,255,0.08); }
    .summary.safe { background: linear-gradient(135deg, rgba(38, 166, 154, 0.26), rgba(20, 42, 53, 0.9)); }
    .summary.danger { background: linear-gradient(135deg, rgba(216, 76, 69, 0.26), rgba(53, 19, 28, 0.96)); }
    .kicker { font-size: 0.82rem; letter-spacing: 0.18em; text-transform: uppercase; opacity: 0.85; }
    .severity { font-size: 1.4rem; margin-top: 0.35rem; font-weight: 700; }
    .active { margin-top: 0.45rem; color: #dbe5ff; }
    .empty-state { padding: 18px 20px; border-radius: 16px; border: 1px dashed rgba(255,255,255,0.16); color: #cdd7f2; }
    .note { margin-top: 0.5rem; color: #9bb0dd; font-size: 0.95rem; }
    table { width: 100%; border-collapse: collapse; color: #f5f7ff; }
    th, td { border-bottom: 1px solid rgba(255,255,255,0.09); padding: 10px 8px; text-align: left; }
    th { color: #8dd7ff; font-weight: 700; }
    pre { white-space: pre-wrap; word-break: break-word; background: rgba(255,255,255,0.05); padding: 16px; border-radius: 14px; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    "<div class='title-wrap'><h1>ValoTox Live UI</h1><p>Paste a Valorant comment and inspect the model output instantly.</p></div>"
                )
                text = gr.Textbox(
                    label="Comment",
                    placeholder="Type something like: ggez bot uninstall",
                    lines=6,
                )
                threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Threshold")
                run_btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                summary = gr.HTML()
                score_table = gr.HTML()
                raw = gr.HTML()
                note = gr.HTML()

        run_btn.click(
            fn=analyze_text,
            inputs=[text, threshold],
            outputs=[summary, score_table, raw, note],
        )
        text.submit(
            fn=analyze_text,
            inputs=[text, threshold],
            outputs=[summary, score_table, raw, note],
        )

    return app


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio UI."""
    app = build_app()
    app.queue(default_concurrency_limit=4)
    app.launch(server_name=host, server_port=port, share=share)