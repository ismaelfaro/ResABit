"""Push an exported checkpoint to the Hugging Face Hub. **You run this, not the agent.**

    python upload_to_hf.py checkpoints/tridi-qwen1.5-0.5b-1bit --repo <user>/tridi-qwen1.5-0.5b-1bit
    python upload_to_hf.py ... --private
    python upload_to_hf.py ... --confirm      # actually upload

Publishing is irreversible in the way that matters: a model page is indexed,
mirrored and quoted long after it is deleted, and a 1-bit checkpoint invites
exactly one misreading -- that 285 perplexity is a working model. So the
default here is a dry run that prints what would be pushed and stops.
Uploading requires `--confirm` and a token you supply yourself.

Preflight refuses to publish a checkpoint whose card was not generated from
its own manifest, because a card describing a different run is worse than no
card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = ("model.safetensors", "config.json", "README.md")


def preflight(directory: Path) -> dict:
    """Refuse anything that would publish a number no run produced."""
    problems = []

    for name in REQUIRED:
        if not (directory / name).exists():
            problems.append(
                f"missing {name}"
                + (
                    "  (run: python make_model_card.py <dir>)"
                    if name == "README.md"
                    else ""
                )
            )
    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))

    manifest = json.loads((directory / "config.json").read_text())
    card = (directory / "README.md").read_text()

    metrics = manifest.get("metrics", {})
    frozen = metrics.get("wikitext2_val_frozen") or metrics.get(
        "wikitext2_val_train_forward"
    )
    if not frozen:
        problems.append("manifest carries no evaluated perplexity")
    else:
        # The card must quote this checkpoint's own frozen perplexity. If it
        # was generated from a different manifest -- an easy mistake when two
        # arms sit in sibling directories -- the number simply will not be in
        # the text.
        headline = f"{frozen['perplexity']:.3f}"
        if headline not in card:
            problems.append(
                f"card does not quote this checkpoint's perplexity ({headline}); "
                f"regenerate it with make_model_card.py"
            )

    if manifest.get("diverged"):
        problems.append("this run is marked diverged in its manifest")

    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--repo", required=True, help="<user-or-org>/<name> on the Hub")
    p.add_argument("--private", action="store_true",
                   help="create the repo private; you can flip it later")
    p.add_argument("--confirm", action="store_true",
                   help="perform the upload (without this it is a dry run)")
    args = p.parse_args()

    directory = Path(args.checkpoint)
    manifest = preflight(directory)

    files = sorted(
        f for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    total = sum(f.stat().st_size for f in files)

    print(f"repository : {args.repo} ({'private' if args.private else 'public'})")
    print(f"source     : {directory}")
    print(f"arm/seed   : {manifest.get('arm')} / {manifest.get('seed')}")
    print(f"commit     : {manifest.get('commit')}")
    metrics = manifest["metrics"]
    frozen = metrics.get("wikitext2_val_frozen")
    if frozen:
        print(f"frozen ppl : {frozen['perplexity']:.3f}  "
              f"(training forward: "
              f"{metrics['wikitext2_val_train_forward']['perplexity']:.3f})")
    print(f"\n{len(files)} files, {total/1e6:.1f} MB total:")
    for f in files:
        print(f"  {f.stat().st_size/1e6:>8.1f} MB  {f.name}")

    if not args.confirm:
        print(
            "\nDRY RUN — nothing uploaded.\n"
            "Read the card once more (it is the part people quote), then rerun "
            "with --confirm."
        )
        sys.exit(0)

    from huggingface_hub import HfApi

    api = HfApi()
    whoami = api.whoami()          # fails early and clearly if no token
    print(f"\nuploading as {whoami['name']} ...")

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(directory),
        repo_id=args.repo,
        repo_type="model",
        commit_message=(
            f"TriDi {manifest.get('arm')} seed {manifest.get('seed')} "
            f"(source commit {manifest.get('commit')})"
        ),
    )
    print(f"done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
