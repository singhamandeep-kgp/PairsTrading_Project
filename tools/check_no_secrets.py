#!/usr/bin/env python
"""
Reject any tracked file containing a known-forbidden identifier.

DESIGN NOTE — why digests and not literals
------------------------------------------
The obvious implementation greps for the real hostname and username. That would
commit the real hostname and username *into the checker*, which is precisely the
bug it exists to prevent.

So this stores SHA-256 digests only. It hashes every token in every file and
compares. Nothing sensitive is ever written down.

The digest of the ROTATED password stays in the list on purpose: a forgotten copy
in an old notebook or scratch file is still caught after the rotation.

Add a digest with:
    python tools/check_no_secrets.py --hash "the-value"

Usage (pre-commit passes filenames):
    python tools/check_no_secrets.py FILE [FILE ...]
"""

from __future__ import annotations

import hashlib
import re
import sys

# SHA-256 of values that must never appear in a tracked file.
# Regenerate with --hash. Never replace these with the plaintext.
FORBIDDEN_SHA256: set[str] = {
    # FactSet SQL Server host
    "c6c9ec9e60f8824272bce7c05870f6642bcfab8b8d80e50d4a165d3a50ab7a1a",
    # SQL username
    "fe7dee8fc7c67685f43a05a3caa5743c5131848d3a4961d3ce82ca731d3f0771",
    # The password that was exposed in a docstring. KEPT AFTER ROTATION on
    # purpose: a forgotten copy in an old notebook or scratch file is still
    # caught once the live credential has changed.
    "56f9a57fd313b0e8dcaf916de0af896d9bc4bbb511f6e0fa444c6de819674dad",
}

# Tokens worth hashing: anything long enough to be a credential or a hostname.
TOKEN_RE = re.compile(r"[A-Za-z0-9@._$!%*#?&:+/=-]{6,}")
DOTTED_QUAD_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

# A NEW credential's value cannot be known in advance, so the shape is matched
# instead. Two narrow patterns, chosen to avoid false positives on ordinary code:
#
#   1. a QUOTED literal assigned to a secret-ish name  ->  PASSWORD = "abc123"
#   2. a .env-style line                               ->  FACTSET_SQL_PASSWORD=abc123
#
# Both deliberately exclude the things that look similar but are fine:
#   password: str | None = None      (a type annotation)
#   password=password                (passing a parameter through)
#   pwd = os.environ.get("...")      (reading from the environment)
#   f"UID={USER};PWD={pwd};"         (interpolating into a connection string)
# Matching those was the first version's mistake: a checker that cries wolf on
# correct code gets disabled, and then it protects nothing.
QUOTED_SECRET_RE = re.compile(
    r"""(?:password|pwd|secret|token|api_key)\s*[:=]\s*(?P<q>["'])(?P<val>[^"']{3,})(?P=q)""",
    re.IGNORECASE,
)
ENV_STYLE_SECRET_RE = re.compile(
    r"^\s*[A-Z0-9_]*(?:PASSWORD|PWD|SECRET|TOKEN|API_KEY)[A-Z0-9_]*\s*=\s*(?P<val>\S+)\s*$",
)
PLACEHOLDER_RE = re.compile(
    r"^(?:<[^>]*>|\{\{.*\}\}|\$\{?\w+\}?|%\w+%|x{3,}|\*{3,}|\.{3,}|"
    r"your[_-]?\w*|my[_-]?\w*|placeholder|redacted|changeme|example|dummy|"
    r"none|null|true|false|str|int|bool)$",
    re.IGNORECASE,
)

SKIP_SUFFIXES = {".parquet", ".pkl", ".png", ".jpg", ".gif", ".pdf", ".zip",
                 ".whl", ".so", ".pyd", ".dll", ".lock"}

# An explicit, per-line opt-out. Needed because some fake secrets are the POINT:
# a test asserting "safe_connection_summary() never leaks the password" has to
# supply a password to not-leak.
#
# Inline and visible, rather than a blanket exemption for tests/ -- a whole
# directory carved out is where a real credential eventually hides. Reviewing this
# marker is cheap; reviewing an exempted directory is not.
ALLOWLIST_MARKER = "pragma: allowlist secret"


def _sha(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def check_file(path: str) -> list[str]:
    if any(path.lower().endswith(s) for s in SKIP_SUFFIXES):
        return []
    # This checker holds the digests, so scanning it would be circular.
    if path.replace("\\", "/").endswith("tools/check_no_secrets.py"):
        return []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    problems = []
    for n, line in enumerate(lines, 1):
        if ALLOWLIST_MARKER in line:
            continue

        for token in set(TOKEN_RE.findall(line)) | set(DOTTED_QUAD_RE.findall(line)):
            if _sha(token) in FORBIDDEN_SHA256:
                problems.append(
                    f"{path}:{n}: contains a forbidden identifier "
                    f"(matched a known digest). Move it to .env."
                )
                break

        stripped = line.lstrip()
        if stripped.startswith(("#", "//", "--")):
            continue  # a commented-out placeholder is the documented pattern

        for rx in (QUOTED_SECRET_RE, ENV_STYLE_SECRET_RE):
            m = rx.search(line)
            if not m:
                continue
            val = m.group("val").strip().rstrip(",;)")
            if val and not PLACEHOLDER_RE.match(val):
                problems.append(
                    f"{path}:{n}: a secret appears to be assigned a literal value "
                    f"({val[:3]}...). Use a placeholder and put the real value in .env."
                )
            break
    return problems


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--hash":
        if len(argv) < 2:
            print("usage: check_no_secrets.py --hash VALUE", file=sys.stderr)
            return 2
        print(_sha(argv[1]))
        return 0

    problems = [p for path in argv for p in check_file(path)]
    if problems:
        print("Blocked: forbidden identifiers found.\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nCredentials and hostnames belong in .env (gitignored), never in a "
            "tracked file. See .env.example.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
