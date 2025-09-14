#!/usr/bin/env python3
"""
Upload RM checkpoints to a single Hugging Face repo as immutable tags.

- Each checkpoint directory (checkpoint-<N>) becomes its own commit, tagged "step-<N>"
- The default branch "main" is pointed at your BEST_STEP (recommended), and we tag:
    - "best"  -> step-<BEST_STEP>
    - "final" -> last chronological checkpoint

Why this shape?
- Clean, immutable "refs" (tags) for every step
- Fast cloning/pushing (no LFS smudge, shallow + blobless)
- Consumers can load: revision="step-12345" / "best" / "final"

Prereqs:
- `huggingface-cli login` on this node (or set HF_TOKEN)
- Repo exists on the Hub (this script can create it if you wish)

Usage:
    python hf_ckpt_tags_uploader.py
"""

import os
import re
import sys
import fnmatch
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import argparse
from xml.parsers.expat import model

# Only copy files matching INCLUDE_PATTERNS (None = allow all)
INCLUDE_PATTERNS = None
# Always exclude these patterns (applied after include filter)
EXCLUDE_PATTERNS = [
    "*.tmp", "*.lock", "*.log", "events.out.tfevents*", "wandb/*", "tensorboard/*", "*.png", "*.jpg"
]

# Track these via Git LFS
LFS_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.ckpt", "*.model", "*.onnx", "*.gguf"]

DRY_RUN                 = False
CREATE_REPO_IF_MISSING  = True   # create Hub repo if it doesn't exist
PRIVATE_REPO            = True   # only used if creating repo
# --------------------------------------


# --- Global env: disable LFS smudge everywhere we shell out ---
LFS_ENV = os.environ.copy()
LFS_ENV["GIT_LFS_SKIP_SMUDGE"] = "1"   # do not auto-download LFS blobs on checkout

def run(cmd, cwd=None, env=LFS_ENV, check=True, capture=False):
    print(">>", " ".join(cmd))
    if capture:
        return subprocess.run(cmd, cwd=cwd, env=env, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)

def parse_step_from_dirname(name: str):
    m = re.search(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None

def list_checkpoints(root: Path):
    if not root.exists():
        sys.exit(f"Checkpoint root not found: {root.resolve()}")
    items = []
    for p in sorted(root.glob("checkpoint-*")):
        if p.is_dir():
            step = parse_step_from_dirname(p.name)
            if step is not None:
                items.append((step, p))
    if not items:
        sys.exit(f"No checkpoint-* folders found under {root}")
    items.sort(key=lambda x: x[0])  # numeric order
    return items

def should_include(rel_path: str) -> bool:
    # INCLUDE (if set) must match at least one pattern
    if INCLUDE_PATTERNS:
        if not any(fnmatch.fnmatch(rel_path, pat) for pat in INCLUDE_PATTERNS):
            return False
    # EXCLUDE
    for pat in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(rel_path, pat):
            return False
    return True

def copy_selected(src_dir: Path, dst_dir: Path):
    src_dir = Path(src_dir)
    for root, dirs, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        # sanitize for top-level
        rel_root = "" if rel_root == "." else rel_root
        # Filter out excluded directories early to avoid walking them
        dirs[:] = [d for d in dirs if should_include(os.path.join(rel_root, d) + "/")]
        for f in files:
            rel_file = os.path.join(rel_root, f) if rel_root else f
            if not should_include(rel_file):
                continue
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dst_dir, rel_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

def clean_worktree(repo_dir: Path):
    for item in repo_dir.iterdir():
        if item.name == ".git":
            continue
        if item.is_file() or item.is_symlink():
            item.unlink()
        else:
            shutil.rmtree(item)

def write_gitattributes(repo_dir: Path):
    gattr = repo_dir / ".gitattributes"
    lines = [f"{pat} filter=lfs diff=lfs merge=lfs -text\n" for pat in LFS_PATTERNS]
    existing = gattr.read_text() if gattr.exists() else ""
    new_content = "".join(lines)
    if existing != new_content:
        gattr.write_text(new_content)
        run(["git", "add", ".gitattributes"], cwd=repo_dir)
        # commit may be a no-op if nothing changed
        subprocess.run(["git", "commit", "-m", "chore: set/update LFS tracking"], cwd=repo_dir, env=LFS_ENV)

def get_existing_tags(repo_dir: Path):
    run(["git", "fetch", "--tags", "--force"], cwd=repo_dir)
    res = run(["git", "tag"], cwd=repo_dir, capture=True, check=True)
    tags = set(res.stdout.split())
    return tags

def ensure_repo_exists(repo_id):
    if not CREATE_REPO_IF_MISSING:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # will raise if not found
        api.repo_info(repo_id, repo_type="model")
        print(f"✓ Repo exists: {repo_id}")
    except Exception:
        print(f"Repo {repo_id} not found; creating...")
        from huggingface_hub import HfApi
        HfApi().create_repo(
            repo_id, private=PRIVATE_REPO, repo_type="model", exist_ok=True
        )
        print("✓ Created.")

def main(model_name):
    # --------------- CONFIG ----------------
    REPO_ID        = f"Oxford-HIPlab/{model_name}"     # One Hub repo for all checkpoints
    CKPT_ROOT      = f"save_reward_models/{model_name}"  # Contains checkpoint-*
    TOKENIZER_SRC  = None      # Optional dir to include as "tokenizer/" in every revision

    # Validate tokenizer path if provided
    if TOKENIZER_SRC:
        tok = Path(TOKENIZER_SRC)
        if not tok.exists():
            sys.exit(f"TOKENIZER_SRC not found: {tok}")

    ckpt_root = Path(CKPT_ROOT)
    checkpoints = list_checkpoints(ckpt_root)
    last_step = checkpoints[-1][0]
    print("✓ Found checkpoints:")
    for step, ckpt_path in checkpoints:
        print(f" - {ckpt_path.name} (step {step})")
    print(f"Last checkpoint step: {last_step}")

    # Get best_step by going into checkpoints[-1] and finding trainer_state.json, then best_model_checkpoint (which is a full path)
    trainer_state_path = checkpoints[-1][1] / "trainer_state.json"
    if trainer_state_path.exists():
        trainer_state = json.loads(trainer_state_path.read_text())
        best_step_file = trainer_state.get("best_model_checkpoint", last_step)
        best_step = parse_step_from_dirname(os.path.basename(best_step_file))
        print(f"Best checkpoint step from final trainer_state.json: {best_step}")
    else:
        best_step = last_step
        print(f"Best checkpoint not found; using last step: {best_step}")

    ensure_repo_exists(REPO_ID)

    with TemporaryDirectory(prefix="hf_ckpt_repo_") as tmpdir:
        tmp = Path(tmpdir)
        print(f"Working in: {tmp}")

        # Shallow, no checkout, blobless clone (prevents LFS smudge & huge downloads)
        run([
            "git", "clone",
            "--depth=1",
            "--no-checkout",
            "--filter=blob:none",
            f"git@hf.co:{REPO_ID}",
            str(tmp)
        ])

        # Ensure LFS doesn't smudge in this repo
        run(["git", "lfs", "install", "--skip-smudge", "--local"], cwd=tmp)
        run(["git", "config", "lfs.fetchexclude", "*"], cwd=tmp)

        # Make sure LFS tracking is set (idempotent)
        write_gitattributes(tmp)

        existing = get_existing_tags(tmp)

        for step, ckpt_path in checkpoints:
            tag = f"step-{step}"
            if tag in existing:
                print(f"✓ Skip {ckpt_path.name}: tag {tag} already exists")
                continue

            print(f"\n=== Processing {ckpt_path.name} -> {tag} ===")
            # Orphan commit with ONLY this checkpoint (plus tokenizer if provided)
            run(["git", "checkout", "--orphan", f"_build-{tag}"], cwd=tmp)
            clean_worktree(tmp)

            # Copy in checkpoint files (filtered)
            copy_selected(ckpt_path, tmp)

            # Optional tokenizer to make the revision self-contained
            if TOKENIZER_SRC:
                copy_selected(Path(TOKENIZER_SRC), tmp / "tokenizer")

            # Minimal README
            (tmp / "README.md").write_text(
                f"# GRM Gemma2 2B — checkpoint {step}\n\n"
                f"This revision contains the files from `{ckpt_path.name}`.\n\n"
                f"Load via:\n\n"
                f"```python\n"
                f"from huggingface_hub import snapshot_download\n"
                f"snapshot_download('{REPO_ID}', revision='step-{step}')\n"
                f"```\n"
            )

            # Ensure .gitattributes exists in this tree
            write_gitattributes(tmp)

            run(["git", "add", "-A"], cwd=tmp)
            # Commit (may be empty if filters excluded everything—guard that)
            commit = subprocess.run(["git", "commit", "-m", f"Add checkpoint {step}"], cwd=tmp, env=LFS_ENV)
            if commit.returncode != 0:
                print(f"⚠️  Nothing to commit for {ckpt_path.name}; skipping tag.")
                # Return to a safe branch before continuing
                run(["git", "checkout", "--detach"], cwd=tmp, check=False)
                continue

            # Create and push tag
            run(["git", "tag", tag], cwd=tmp)
            if not DRY_RUN:
                run(["git", "push", "origin", tag], cwd=tmp)
            else:
                print(f"(dry-run) would push tag {tag}")

            # Detach to keep workspace clean for next orphan
            run(["git", "checkout", "--detach"], cwd=tmp, check=False)

        # Point main at 'best' and set convenience tags
        print("\n=== Updating main/best/final ===")
        run(["git", "fetch", "--tags", "--force"], cwd=tmp)
        run(["git", "checkout", "-B", "main", f"step-{best_step}"], cwd=tmp)
        if not DRY_RUN:
            run(["git", "push", "origin", "main", "--force"], cwd=tmp)
        else:
            print(f"(dry-run) would force-push main -> step-{best_step}")

        # Move/refresh tags 'best' and 'final'
        for name, target in [("best", f"step-{best_step}"), ("final", f"step-{last_step}")]:
            # Replace local tag
            subprocess.run(["git", "tag", "-d", name], cwd=tmp, env=LFS_ENV)
            run(["git", "tag", name, target], cwd=tmp)
            if not DRY_RUN:
                # Replace remote tag
                run(["git", "push", "origin", f":refs/tags/{name}"], cwd=tmp, check=False)  # delete if exists
                run(["git", "push", "origin", name, "--force"], cwd=tmp)
            else:
                print(f"(dry-run) would set {name} -> {target}")

        print("\n🎉 Done. Revisions available:")
        print(f"- main  -> step-{best_step} (recommended default)")
        print(f"- best  -> step-{best_step}")
        print(f"- final -> step-{last_step}")
        print(f"- step-<N> for each uploaded checkpoint")
        print("\nExamples:")
        print(f"  from transformers import AutoModel\n"
              f"  model = AutoModel.from_pretrained('{REPO_ID}')                 # best\n"
              f"  model = AutoModel.from_pretrained('{REPO_ID}', revision='step-{best_step}')\n")

if __name__ == "__main__":
    # get --model from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name")
    args = parser.parse_args()
    try:
        main(args.model)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e}\n")
        sys.exit(1)
