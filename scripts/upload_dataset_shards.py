"""Explicit upload to an existing/new HF dataset repository; uses HF_TOKEN."""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument(
        "--public", action="store_true", help="Default is private; respect dataset redistribution terms"
    )
    args = p.parse_args()
    path = Path(args.manifest)
    data = json.loads(path.read_text())
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    for shard in data["shards"]:
        local = path.parent / shard["url"]
        api.upload_file(
            path_or_fileobj=str(local), path_in_repo=local.name, repo_id=args.repo_id, repo_type="dataset"
        )
        shard["url"] = f"hf://datasets/{args.repo_id}/{local.name}"
    remote_manifest = path.with_name("shards.remote.json")
    remote_manifest.write_text(json.dumps(data, indent=2))
    api.upload_file(
        path_or_fileobj=str(remote_manifest),
        path_in_repo="shards.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(remote_manifest)


if __name__ == "__main__":
    main()
