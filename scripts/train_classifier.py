"""Lab 3A starter: fine-tune the Bayan topic classifier."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact (local path or mounted Drive path).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO(Lab 3A): load grouped dataset, tokenizer/checkpoint from Lab 1 decision,
    # fine-tune, evaluate and save a re-runnable artefact into output_dir.
    raise NotImplementedError(
        f"Complete Lab 3A classifier training; output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()
