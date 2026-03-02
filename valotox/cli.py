"""
ValoTox CLI — unified entry point for all pipeline phases.

Usage:
    python -m valotox.cli scrape    --source reddit
    python -m valotox.cli merge
    python -m valotox.cli annotate  --create-project
    python -m valotox.cli iaa
    python -m valotox.cli eda
    python -m valotox.cli train     --models roberta hatebert
    python -m valotox.cli serve
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        prog="valotox",
        description="ValoTox — Valorant Toxicity Detection System CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline phase to run")

    # ── scrape ───────────────────────────────────────────────────────────
    scrape_p = sub.add_parser("scrape", help="Scrape data from sources")
    scrape_p.add_argument(
        "--source",
        choices=["reddit", "vlr", "twitter", "all"],
        default="all",
    )
    scrape_p.add_argument("--posts", type=int, default=500)
    scrape_p.add_argument("--pages", type=int, default=20)

    # ── merge ────────────────────────────────────────────────────────────
    merge_p = sub.add_parser("merge", help="Merge & prepare raw data")
    merge_p.add_argument("--split", action="store_true", help="Also create annotation splits")
    merge_p.add_argument("--iaa-size", type=int, default=600)
    merge_p.add_argument("--total-size", type=int, default=10000)

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

    # ── train ────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train & benchmark models")
    train_p.add_argument("--input", type=str, default=None)
    train_p.add_argument("--models", nargs="+", default=None)
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--jigsaw", action="store_true")

    # ── serve ────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start FastAPI server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # ── Dispatch ─────────────────────────────────────────────────────────
    if args.command == "scrape":
        if args.source in ("reddit", "all"):
            from valotox.scraping.reddit_scraper import scrape_subreddits

            scrape_subreddits(posts_per_sort=args.posts)
        if args.source in ("vlr", "all"):
            from valotox.scraping.vlr_scraper import scrape_vlr

            scrape_vlr(pages=args.pages)
        if args.source in ("twitter", "all"):
            from valotox.scraping.twitter_scraper import scrape_twitter

            scrape_twitter()

    elif args.command == "merge":
        from valotox.scraping.data_pipeline import create_annotation_splits, merge_sources

        merged = merge_sources()
        if args.split:
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

        run_full_eda(args.input)

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

    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "valotox.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
