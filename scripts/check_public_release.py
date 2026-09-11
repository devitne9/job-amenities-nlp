#!/usr/bin/env python3
"""Check public Git content without reading ignored data or printing matched values.

This is a conservative release guard, not a substitute for reviewing source text,
figures, and staged diffs. Run with --staged to inspect index blobs, or --history to
inspect every commit reachable from local Git refs.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000
ALLOWED_NAMES = {"README.md", "LICENSE", ".gitignore", "requirements.txt"}
ALLOWED_FIGURES = {
    "figures/coefficient_estimates.png",
    "figures/skill_group_slopes.png",
}
PATTERNS = {
    "credential token": re.compile(
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}"
        r"|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
        r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    ),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|passwd|secret|authorization|cookie)"
        r"\s*[=:]\s*[\"'][^\"'\n]{6,}[\"']"
    ),
    "personal absolute path": re.compile(
        r"/(?:Users|home)/[^\s\"']+|[A-Za-z]:[\\/]+Users[\\/]+[^\s\"']+"
        r"|(?:Desktop|Downloads)/[^\s\"']+"
    ),
    "database connection URL": re.compile(
        r"(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
    ),
    "URL with credentials": re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def allowed_path(name: str) -> bool:
    path = PurePosixPath(name)
    if name in ALLOWED_NAMES or name in ALLOWED_FIGURES:
        return True
    if len(path.parts) != 2:
        return False
    return (path.parts[0] in {"src", "scripts"} and path.suffix == ".py") or (
        path.parts[0] == "docs" and path.suffix == ".md"
    )


def check_png(data: bytes) -> list[str]:
    """Reject unexpected PNG metadata or trailing payloads; visuals need review."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ["invalid PNG signature"]
    offset = 8
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        if offset > len(data):
            return ["truncated PNG chunk"]
        if kind == b"tEXt":
            key, separator, value = payload.partition(b"\0")
            if not separator or key != b"Software" or not value.startswith(b"Matplotlib version"):
                return ["PNG text metadata requires review"]
        elif kind not in {b"IHDR", b"IDAT", b"IEND", b"pHYs"}:
            return ["PNG ancillary content requires review"]
        if kind == b"IEND":
            return [] if offset == len(data) else ["PNG trailing content requires review"]
    return ["missing PNG end chunk"]


def check_content(name: str, data: bytes) -> list[str]:
    """Return reason labels only, never a matched credential or private text."""
    issues = []
    if not allowed_path(name):
        issues.append("file outside reviewed public types/locations")
    if len(data) > MAX_BYTES:
        issues.append("file exceeds 1 MB review threshold")
    if name in ALLOWED_FIGURES:
        issues.extend(check_png(data))
        return issues
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return issues + ["unexpected binary content"]
    for label, pattern in PATTERNS.items():
        if pattern.search(content):
            issues.append(label)
    if name.endswith(".py"):
        try:
            ast.parse(content, filename=name)
        except SyntaxError:
            issues.append("Python syntax error")
    return issues


def entries(staged: bool, history: bool):
    if history:
        seen = set()
        for commit in git("rev-list", "--all").decode().splitlines():
            for record in git("ls-tree", "-rz", commit).split(b"\0"):
                if not record:
                    continue
                header, raw_name = record.split(b"\t", 1)
                mode, kind, oid = header.decode().split()
                name = raw_name.decode()
                if (name, oid, mode) in seen:
                    continue
                seen.add((name, oid, mode))
                data = git("cat-file", "blob", oid) if kind == "blob" else b""
                yield name, data, mode
        return
    if staged:
        for record in git("ls-files", "--stage", "-z").split(b"\0"):
            if record:
                header, raw_name = record.split(b"\t", 1)
                mode, oid, stage = header.decode().split()
                if stage != "0":
                    raise ValueError("Unresolved Git index conflict")
                yield raw_name.decode(), git("cat-file", "blob", oid), mode
        return
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for name in sorted(set(names.decode().split("\0")) - {""}):
        path = ROOT / name
        if path.is_symlink():
            yield name, b"", "120000"
        elif path.is_file():
            yield name, path.read_bytes(), "100644"
        else:
            raise ValueError("Tracked file is missing; review Git status")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--history", action="store_true")
    args = parser.parse_args()
    count = 0
    failed = False
    for name, data, mode in entries(args.staged, args.history):
        count += 1
        issues = check_content(name, data)
        if mode not in {"100644", "100755"}:
            issues.append("symlink or non-regular Git entry")
        if issues:
            failed = True
            print(f"FAIL: {name}: {'; '.join(issues)}")
    if not count:
        print("FAIL: no public files inspected")
        return 1
    if failed:
        return 1
    print(f"PASS: {count} public file versions checked; manual review is still required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
