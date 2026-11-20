"""在 github.com:443 不可达时，用 GitHub Git Data API 把本地提交历史整体推上去。

用法（token 只放环境变量，不要写进命令历史）：
    $env:GITHUB_TOKEN="ghp_xxx"     # Linux/macOS: export GITHUB_TOKEN=ghp_xxx
    python scripts/github_api_push.py --repo Hs-happy/ai-shopping-agent --branch main

做法：逐个本地 commit → 上传 blob（按 SHA 去重）→ 自底向上拼 tree → 建 commit，
每步都更新 ref 让对象保持可达。仓库必须已存在；空仓库会先落一个占位提交解锁 blob 接口。
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import time
from typing import Any

import httpx

API = "https://api.github.com"


def git(*args: str, text: bool = True) -> str | bytes:
    out = subprocess.run(["git", *args], check=True, capture_output=True)
    return out.stdout.decode("utf-8", "replace") if text else out.stdout


def local_commits() -> list[dict[str, str]]:
    # 用 \x1f 分隔字段、\x1e 分隔记录：根提交没有父提交，%P 为空，
    # 若用连续的 \x00\x00 当分隔符会把 "sha\x00\x00" 误判成记录边界。
    fmt = "%H%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%B%x1e"
    raw = str(git("log", "--reverse", f"--format={fmt}"))
    commits = []
    for chunk in raw.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        sha, parents, author, email, date, message = chunk.split("\x1f", 5)
        commits.append(
            {"sha": sha, "parents": parents.split(), "author": author, "email": email,
             "date": date, "message": message.strip("\n")}
        )
    return commits


class Pusher:
    def __init__(self, repo: str, branch: str, token: str):
        self.repo = repo
        self.branch = branch
        self.client = httpx.Client(
            base_url=f"{API}/repos/{repo}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "User-Agent": "dsh-agent-push"},
            timeout=90,
        )
        self.blob_cache: dict[str, str] = {}
        self.tree_cache: dict[tuple[str, ...], str] = {}

    # ------------------------------------------------------------------ http
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.client.post(path, json=payload)
        if resp.status_code >= 300:
            print(f"  ! POST {path} -> {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ git data
    def blob(self, sha: str) -> str:
        if sha in self.blob_cache:
            return self.blob_cache[sha]
        content = git("cat-file", "blob", sha, text=False)
        data = self._post("/git/blobs", {
            "content": base64.b64encode(content).decode("ascii"), "encoding": "base64",
        })
        self.blob_cache[sha] = data["sha"]
        return data["sha"]

    def tree(self, files: dict[str, str], prefix: tuple[str, ...] = ()) -> str:
        """files: {path: blob_sha}，递归按目录组装 tree（GitHub 的 tree 接口不认带 / 的 path）。"""
        entries: list[dict[str, Any]] = []
        dirs: dict[str, dict[str, str]] = {}
        for path, blob_sha in files.items():
            parts = path.split("/")
            if len(parts) == 1:
                entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob_sha})
            else:
                dirs.setdefault(parts[0], {})[ "/".join(parts[1:]) ] = blob_sha
        for name, sub in dirs.items():
            key = prefix + (name,)
            if key not in self.tree_cache:
                self.tree_cache[key] = self.tree(sub, key)
            entries.append({"path": name, "mode": "040000", "type": "tree", "sha": self.tree_cache[key]})
        return self._post("/git/trees", {"tree": entries})["sha"]

    def files_of(self, commit_sha: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for line in str(git("ls-tree", "-r", commit_sha)).splitlines():
            meta, path = line.split("\t", 1)
            _mode, _type, blob_sha = meta.split()
            files[path] = blob_sha
        return files

    # ------------------------------------------------------------------ ref
    def ref_sha(self) -> str | None:
        resp = self.client.get(f"/git/ref/heads/{self.branch}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()["object"]["sha"]

    def ensure_non_empty(self) -> None:
        """空仓库上创建 blob 会返回 409，先用 Contents API 落一个占位提交解锁。"""
        if self.ref_sha():
            return
        print("仓库为空，先创建占位提交…")
        resp = self.client.put("/contents/.github-init", json={
            "message": "chore: bootstrap (temporary)",
            "content": base64.b64encode(b"init\n").decode("ascii"),
        })
        resp.raise_for_status()

    def set_ref(self, commit_sha: str) -> bool:
        """把分支指向 commit_sha；GitHub 偶发 404 但实际已生效，所以以读回结果为准。"""
        for _ in range(3):
            if self.ref_sha() == commit_sha:
                return True
            if self.ref_sha():
                resp = self.client.patch(f"/git/refs/heads/{self.branch}", json={"sha": commit_sha, "force": True})
            else:
                resp = self.client.post("/git/refs", json={"ref": f"refs/heads/{self.branch}", "sha": commit_sha})
            if resp.status_code < 300 and self.ref_sha() == commit_sha:
                return True
            if resp.status_code >= 300:
                print(f"  ! ref update -> {resp.status_code}: {resp.text[:120]} (retry)")
            time.sleep(1.5)
        return self.ref_sha() == commit_sha

    def remote_chain(self) -> list[dict[str, Any]]:
        """返回当前分支上的远端提交（从旧到新），用于断点续传。"""
        resp = self.client.get("/commits", params={"sha": self.branch, "per_page": 100})
        if resp.status_code != 200:
            return []
        items = list(reversed(resp.json()))
        return [{"sha": it["sha"], "message": it["commit"]["message"]} for it in items]

    # ------------------------------------------------------------------ main
    def push(self) -> str:
        self.ensure_non_empty()
        commits = local_commits()
        chain = self.remote_chain()
        matched = 0
        while (matched < len(chain) and matched < len(commits)
               and chain[matched]["message"].strip() == commits[matched]["message"].strip()):
            matched += 1
        remote_sha = chain[matched - 1]["sha"] if matched else None
        if matched:
            print(f"检测到远端已有 {matched} 个同内容提交，从第 {matched + 1} 个继续")

        for idx, commit in enumerate(commits[matched:], start=matched + 1):
            files = self.files_of(commit["sha"])
            blobs = {path: self.blob(sha) for path, sha in files.items()}
            tree_sha = self.tree(blobs)
            body: dict[str, Any] = {
                "message": commit["message"],
                "tree": tree_sha,
                "parents": [remote_sha] if remote_sha else [],
                "author": {"name": commit["author"], "email": commit["email"], "date": commit["date"]},
                "committer": {"name": commit["author"], "email": commit["email"], "date": commit["date"]},
            }
            created = self._post("/git/commits", body)["sha"]
            if not self.set_ref(created):
                raise RuntimeError(f"ref 更新失败: {created}")
            remote_sha = created
            print(f"[{idx}/{len(commits)}] {commit['sha'][:8]} -> {created[:8]}  "
                  f"{commit['message'].splitlines()[0][:56]}  ({len(files)} files)")
        return remote_sha or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("请先设置 GITHUB_TOKEN 环境变量")

    head = Pusher(args.repo, args.branch, token).push()
    print(f"\n完成：{args.repo}@{args.branch} -> {head}")


if __name__ == "__main__":
    main()
