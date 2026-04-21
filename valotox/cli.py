"""
ValoTox CLI — unified entry point for all pipeline phases.

Usage:
    python -m valotox.cli scrape    --source reddit
    python -m valotox.cli merge
    python -m valotox.cli annotate  --create-project
    python -m valotox.cli iaa
    python -m valotox.cli eda
    python -m valotox.cli train     --models roberta hatebert
    python -m valotox.cli predict   --input data.csv --text-column utterance
    python -m valotox.cli interactive
    python -m valotox.cli serve
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from valotox.config import ROOT_DIR


def main():
    parser = argparse.ArgumentParser(
        prog="valotox",
        description="ValoTox — Valorant Toxicity Detection System CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline phase to run")

    # ── scrape ───────────────────────────────────────────────────────────
    scrape_p = sub.add_parser("scrape", help="Scrape Reddit data")
    scrape_p.add_argument(
        "--source",
        choices=["reddit"],
        default="reddit",
    )
    scrape_p.add_argument("--posts", type=int, default=500)

    # ── merge ────────────────────────────────────────────────────────────
    merge_p = sub.add_parser("merge", help="Merge & prepare raw data")
    merge_p.add_argument("--split", action="store_true", help="Also create annotation splits")
    merge_p.add_argument("--iaa-size", type=int, default=600)
    merge_p.add_argument("--total-size", type=int, default=10000)
    merge_p.add_argument(
        "--reddit-conda",
        action="store_true",
        help="Merge processed Reddit data with CONDA in-game chat CSV",
    )
    merge_p.add_argument(
        "--conda-csv",
        type=str,
        default=None,
        help="Path to CONDA CSV (utterance column). Default: <repo>/../data/raw/CONDA_test.csv",
    )

    # ── annotate ─────────────────────────────────────────────────────────
    ann_p = sub.add_parser("annotate", help="Annotation pipeline")
    ann_p.add_argument("--create-project", action="store_true")
    ann_p.add_argument("--import-csv", type=str, default=None)
    ann_p.add_argument("--project-id", type=int, default=1)
    ann_p.add_argument("--export", action="store_true")
    ann_p.add_argument("--gpt4o", type=str, default=None, help="CSV for GPT-4o annotation")
    ann_p.add_argument("--gpt4o-batch-size", type=int, default=None)

    # ── iaa ──────────────────────────────────────────────────────────────
    sub.add_parser("iaa", help="Compute inter-annotator agreement")

    # ── eda ──────────────────────────────────────────────────────────────
    eda_p = sub.add_parser("eda", help="Run exploratory data analysis")
    eda_p.add_argument("--input", type=str, default=None)
    eda_p.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="If gold labels are all zero, use pred_*≥this for label plots (default: 0.5)",
    )
    eda_p.add_argument(
        "--no-pred-proxy",
        action="store_true",
        help="Do not fill label plots from pred_* when gold is empty",
    )

    # ── train ────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train & benchmark models")
    train_p.add_argument("--input", type=str, default=None)
    train_p.add_argument("--models", nargs="+", default=None)
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--jigsaw", action="store_true")

    # ── train-report ─────────────────────────────────────────────────────
    report_p = sub.add_parser("train-report", help="Print latest training report table")
    report_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model run directory name under models/ (e.g. roberta-valotox). Default: latest run",
    )

    # ── predict ──────────────────────────────────────────────────────────
    pred_p = sub.add_parser("predict", help="Run toxicity model on a CSV column")
    pred_p.add_argument("--input", type=str, required=True, help="Input CSV path")
    pred_p.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing text to classify (default: text)",
    )
    pred_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <input_stem>_predictions.csv next to input)",
    )
    pred_p.add_argument("--threshold", type=float, default=0.5)
    pred_p.add_argument("--batch-size", type=int, default=64)

    # ── interactive ──────────────────────────────────────────────────────
    int_p = sub.add_parser(
        "interactive",
        help="Type comments in the terminal; print toxicity predictions (model only)",
    )
    int_p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Label score threshold for active_labels (default: 0.5)",
    )
    int_p.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="Classify this string once and exit (no REPL)",
    )

    # ── serve ────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start FastAPI server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    # ── ui ────────────────────────────────────────────────────────────────
    ui_p = sub.add_parser("ui", help="Start the Gradio live UI")
    ui_p.add_argument("--host", default="127.0.0.1")
    ui_p.add_argument("--port", type=int, default=7860)
    ui_p.add_argument("--share", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # ── Dispatch ─────────────────────────────────────────────────────────
    if args.command == "scrape":
        from valotox.scraping.reddit_scraper import scrape_subreddits

        scrape_subreddits(posts_per_sort=args.posts)

    elif args.command == "merge":
        from valotox.scraping.data_pipeline import (
            create_annotation_splits,
            merge_reddit_and_conda,
            merge_sources,
        )

        if args.reddit_conda:
            conda_path = args.conda_csv
            if not conda_path:
                default_conda = ROOT_DIR.parent / "data" / "raw" / "CONDA_test.csv"
                conda_path = str(default_conda) if default_conda.is_file() else None
            if not conda_path:
                logger.error(
                    "CONDA CSV not found. Pass --conda-csv /path/to/CONDA_test.csv "
                    f"(looked for {ROOT_DIR.parent / 'data' / 'raw' / 'CONDA_test.csv'})"
                )
                sys.exit(1)
            merged = merge_reddit_and_conda(conda_path)
        else:
            merged = merge_sources()

        if args.split and merged is not None and not getattr(merged, "empty", True):
            create_annotation_splits(
                df=merged,
                iaa_size=args.iaa_size,
                total_annotation_size=args.total_size,
            )

    elif args.command == "annotate":
        if args.create_project:
            from valotox.annotation.label_studio import create_project

            create_project()
        if args.import_csv:
            from valotox.annotation.label_studio import import_tasks

            import_tasks(args.project_id, args.import_csv)
        if args.export:
            from valotox.annotation.label_studio import export_annotations

            export_annotations(args.project_id)
        if args.gpt4o:
            from valotox.annotation.gpt4o_annotator import annotate_batch

            annotate_batch(args.gpt4o, batch_size=args.gpt4o_batch_size)

    elif args.command == "iaa":
        from valotox.annotation.iaa import iaa_report

        iaa_report()

    elif args.command == "eda":
        from valotox.eda import run_full_eda

        run_full_eda(
            args.input,
            pred_threshold=args.pred_threshold,
            use_prediction_proxy=not args.no_pred_proxy,
        )

    elif args.command == "train":
        from valotox.models.benchmark import evaluate_jigsaw_on_valotox, run_benchmark

        run_benchmark(
            annotated_path=args.input,
            models=args.models,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if args.jigsaw:
            evaluate_jigsaw_on_valotox(args.input)

    elif args.command == "train-report":
        from valotox.models.benchmark import generate_training_report

        print(generate_training_report(model_name=args.model))

    elif args.command == "predict":
        from valotox.models.predict_csv import predict_csv

        predict_csv(
            args.input,
            text_column=args.text_column,
            output_path=args.output,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )

    elif args.command == "interactive":
        from valotox.interactive_cli import run_interactive

        run_interactive(threshold=args.threshold, text=args.text)

    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "valotox.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "ui":
        from valotox.ui import launch_ui

        launch_ui(host=args.host, port=args.port, share=args.share)

    if args.command != "interactive":
        logger.info("Done ✓")


if __name__ == "__main__":
    main()
