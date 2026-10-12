"""Collect Urdu Wikimedia articles through the official MediaWiki API.

The program is sequential, throttled, resumable, and records one provenance
row per saved page. This supplies auditable evidence for the assignment's
self-collection requirement instead of treating a prebuilt dump as scraped data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


END_OF_TEXT = "<|endoftext|>"
PROJECT_HOSTS = {
    "wikipedia": "ur.wikipedia.org",
    "wikisource": "ur.wikisource.org",
}
DEFAULT_AGENT = "UrduBPEAssignment/1.0 (NUST SEECS CS417; educational research)"


def request_page(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    *,
    retries: int = 8,
) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(retries):
        try:
            response = session.get(endpoint, params=params, timeout=60)
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = float(response.headers.get("Retry-After", delay))
                time.sleep(max(delay, retry_after))
                delay = min(delay * 2, 60)
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("error", {}).get("code") == "maxlag":
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            return payload
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable retry state")


def collect(
    project: str,
    output: Path,
    metadata: Path,
    state_path: Path,
    *,
    target_mb: float,
    delay_seconds: float,
    user_agent: str,
) -> None:
    host = PROJECT_HOSTS[project]
    endpoint = f"https://{host}/w/api.php"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    elif output.exists() or metadata.exists():
        raise FileExistsError("output exists without resume state; move it or provide the matching state file")

    target_bytes = int(target_mb * 1024 * 1024) if target_mb > 0 else None
    written_bytes = output.stat().st_size if output.exists() else 0
    documents = int(state.get("documents", 0))
    continuation = state.get("continue", {})
    started_at = state.get("started_at") or datetime.now(timezone.utc).isoformat()

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Api-User-Agent": user_agent})
    common_params: dict[str, Any] = {
        "action": "query",
        "generator": "allpages",
        "gapnamespace": 0,
        "gapfilterredir": "nonredirects",
        "gaplimit": "max",
        "prop": "info|revisions|pageprops",
        "inprop": "url",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
    }

    mode = "a" if output.exists() else "w"
    with output.open(mode, encoding="utf-8", newline="\n") as corpus, metadata.open(
        mode, encoding="utf-8", newline="\n"
    ) as evidence:
        while target_bytes is None or written_bytes < target_bytes:
            params = {**common_params, **continuation}
            payload = request_page(session, endpoint, params)
            if "error" in payload:
                raise RuntimeError(f"MediaWiki API error: {payload['error']}")
            pages = payload.get("query", {}).get("pages", [])
            for page in pages:
                revision = (page.get("revisions") or [{}])[0]
                source_text = revision.get("slots", {}).get("main", {}).get("content", "").strip()
                if len(source_text) < 200 or "disambiguation" in page.get("pageprops", {}):
                    continue
                document = f"{page['title']}\n\n{source_text}"
                serialized = f"{document}\n\n{END_OF_TEXT}\n\n"
                byte_count = len(serialized.encode("utf-8"))
                corpus.write(serialized)
                record = {
                    "project": project,
                    "api_endpoint": endpoint,
                    "page_id": page.get("pageid"),
                    "revision_id": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                    "title": page["title"],
                    "url": page.get("fullurl"),
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "utf8_bytes": byte_count,
                    "sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                }
                evidence.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_bytes += byte_count
                documents += 1
                if target_bytes is not None and written_bytes >= target_bytes:
                    break

            corpus.flush()
            evidence.flush()
            continuation = payload.get("continue", {})
            state = {
                "project": project,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "documents": documents,
                "bytes": written_bytes,
                "continue": continuation,
                "complete": not bool(continuation),
                "user_agent": user_agent,
            }
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{project}: documents={documents:,} raw_mb={written_bytes / 1024 / 1024:.2f}", flush=True)

            if target_bytes is not None and written_bytes >= target_bytes:
                break
            if not continuation:
                break
            time.sleep(delay_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Urdu Wikipedia or Wikisource through MediaWiki API")
    parser.add_argument("--project", choices=sorted(PROJECT_HOSTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--target-mb", type=float, default=0, help="0 means collect every namespace-0 page")
    parser.add_argument("--delay", type=float, default=0.25, help="pause between sequential API requests")
    parser.add_argument("--user-agent", default=DEFAULT_AGENT)
    arguments = parser.parse_args()
    collect(
        arguments.project,
        arguments.output,
        arguments.metadata,
        arguments.state,
        target_mb=arguments.target_mb,
        delay_seconds=arguments.delay,
        user_agent=arguments.user_agent,
    )


if __name__ == "__main__":
    main()
