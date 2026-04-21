"""
Label Studio integration for ValoTox annotation workflow.

- Creates a multi-label toxicity project
- Imports data batches
- Exports annotations to CSV
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from valotox.config import ANNOTATED_DIR, settings
from valotox.lexicon import LABELS

# ── Label Studio XML template ────────────────────────────────────────────────
LABEL_CONFIG_XML = """
<View>
  <Header value="ValoTox — Valorant Toxicity Annotation"/>
  <Text name="text" value="$text"/>
  <Choices name="labels" toName="text" choice="multiple" showInline="true">
    <Choice value="toxic" background="red"/>
    <Choice value="harassment" background="orange"/>
    <Choice value="gender_attack" background="purple"/>
    <Choice value="slur" background="darkred"/>
    <Choice value="passive_toxic" background="goldenrod"/>
    <Choice value="not_toxic" background="green"/>
  </Choices>
  <TextArea name="notes" toName="text" rows="2"
            placeholder="Optional annotator notes…" maxSubmissions="1"/>
</View>
"""


def create_project(project_name: str = "ValoTox Annotation") -> int:
    """Create a Label Studio project and return its ID.

    Requires Label Studio to be running at ``LABEL_STUDIO_URL``
    with a valid ``LABEL_STUDIO_API_KEY``.
    """
    from label_studio_sdk import Client

    ls = Client(url=settings.label_studio_url, api_key=settings.label_studio_api_key)
    project = ls.start_project(
        title=project_name,
        label_config=LABEL_CONFIG_XML,
    )
    logger.info(f"Created Label Studio project: id={project.id}, name='{project_name}'")
    return project.id


def import_tasks(
    project_id: int,
    csv_path: Path | str,
    text_column: str = "text_clean",
) -> int:
    """Import a CSV of texts into a Label Studio project.

    Returns the number of tasks imported.
    """
    from label_studio_sdk import Client

    ls = Client(url=settings.label_studio_url, api_key=settings.label_studio_api_key)
    project = ls.get_project(project_id)

    df = pd.read_csv(csv_path)
    tasks = [{"data": {"text": row[text_column]}} for _, row in df.iterrows() if row[text_column]]

    project.import_tasks(tasks)
    logger.info(f"Imported {len(tasks)} tasks into project {project_id}")
    return len(tasks)


def export_annotations(
    project_id: int,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Export completed annotations from Label Studio to a multi-label CSV.

    Each label column is binary (0/1).
    """
    from label_studio_sdk import Client

    output_path = Path(output_path) if output_path else ANNOTATED_DIR / "valotox_annotated.csv"

    ls = Client(url=settings.label_studio_url, api_key=settings.label_studio_api_key)
    project = ls.get_project(project_id)
    tasks = project.get_labeled_tasks()

    rows: list[dict] = []
    for task in tasks:
        text = task["data"].get("text", "")
        for annotation in task.get("annotations", []):
            label_set: set[str] = set()
            notes = ""
            for result in annotation.get("result", []):
                if result["from_name"] == "labels":
                    label_set.update(result["value"]["choices"])
                elif result["from_name"] == "notes":
                    notes = result["value"].get("text", [""])[0]

            row = {"text": text, "annotator": annotation.get("completed_by", "")}
            for label in LABELS:
                row[label] = 1 if label in label_set else 0
            row["notes"] = notes
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} annotations → {output_path}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Label Studio integration")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("create", help="Create annotation project")
    imp = sub.add_parser("import", help="Import tasks")
    imp.add_argument("--project-id", type=int, required=True)
    imp.add_argument("--csv", type=str, required=True)

    exp = sub.add_parser("export", help="Export annotations")
    exp.add_argument("--project-id", type=int, required=True)

    args = parser.parse_args()

    if args.cmd == "create":
        create_project()
    elif args.cmd == "import":
        import_tasks(args.project_id, args.csv)
    elif args.cmd == "export":
        export_annotations(args.project_id)
