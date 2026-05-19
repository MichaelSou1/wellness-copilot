import argparse
import json
import importlib
import time
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "shared": {
        "title": "WHO - Healthy diet (Fact sheet)",
        "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "output": "who_healthy_diet.md",
    },
    "nutritionist": {
        "title": "USDA - MyPlate (Dietary guidance)",
        "url": "https://www.dietaryguidelines.gov/",
        "fallback_urls": [
            "https://www.myplate.gov/eat-healthy/what-is-myplate",
        ],
        "output": "usda_dietary_guidelines.md",
    },
    "trainer": {
        "title": "WHO - Physical activity (Fact sheet)",
        "url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        "output": "who_physical_activity.md",
    },
    "psychologist": {
        "title": "WHO - Mental health: strengthening our response",
        "url": "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
        "output": "who_mental_health.md",
    },
}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def extract_main_text(html: str) -> str:
    bs4 = importlib.import_module("bs4")
    BeautifulSoup = bs4.BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "footer", "nav"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("section")
        or soup.find("body")
        or soup
    )

    lines = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue

        name = node.name.lower()
        if name == "h1":
            lines.append(f"# {text}")
        elif name == "h2":
            lines.append(f"## {text}")
        elif name == "h3":
            lines.append(f"### {text}")
        elif name == "h4":
            lines.append(f"#### {text}")
        elif name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    cleaned = "\n".join(lines).strip()
    return cleaned


def build_markdown(title: str, url: str, body_text: str) -> str:
    downloaded_at = datetime.now(timezone.utc).isoformat()
    return (
        f"# {title}\n\n"
        f"- Source: {url}\n"
        f"- Downloaded at (UTC): {downloaded_at}\n\n"
        f"---\n\n"
        f"{body_text}\n"
    )


def download_one(target_dir: Path, key: str, timeout: int, force: bool, retries: int) -> dict:
    requests = importlib.import_module("requests")
    config = SOURCES[key]
    primary_url = config["url"]
    candidate_urls = [primary_url, *(config.get("fallback_urls") or [])]
    out_path = target_dir / key / config["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        return {
            "bucket": key,
            "status": "skipped",
            "reason": "file exists (use --force to overwrite)",
            "path": str(out_path),
            "url": primary_url,
        }

    errors = []
    for url in candidate_urls:
        for attempt in range(1, retries + 2):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=timeout)
                resp.raise_for_status()
                body_text = extract_main_text(resp.text)
                if not body_text:
                    body_text = "(No extractable body text found. Please verify source page structure.)"

                markdown = build_markdown(config["title"], url, body_text)
                out_path.write_text(markdown, encoding="utf-8")
                return {
                    "bucket": key,
                    "status": "ok",
                    "path": str(out_path),
                    "url": url,
                    "chars": len(markdown),
                }
            except Exception as e:
                errors.append(f"{url} (attempt {attempt}/{retries + 1}): {e}")
                if attempt <= retries:
                    time.sleep(1.2)

    return {
        "bucket": key,
        "status": "failed",
        "url": primary_url,
        "error": " | ".join(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download authoritative corpus files for each knowledge_base bucket."
    )
    parser.add_argument(
        "--kb-dir",
        default="knowledge_base",
        help="Knowledge base root directory",
    )
    parser.add_argument(
        "--only",
        default="all",
        choices=["all", "shared", "nutritionist", "trainer", "psychologist"],
        help="Download only one bucket or all",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Retries per URL when request fails")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--report",
        default="reports/knowledge_download_report.json",
        help="Output report path",
    )
    args = parser.parse_args()

    kb_root = Path(args.kb_dir)
    kb_root.mkdir(parents=True, exist_ok=True)

    targets = [args.only] if args.only != "all" else list(SOURCES.keys())
    results = [download_one(kb_root, key, args.timeout, args.force, args.retries) for key in targets]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kb_root": str(kb_root.resolve()),
        "targets": targets,
        "results": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Knowledge Download] Completed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[Knowledge Download] Report: {report_path}")


if __name__ == "__main__":
    main()
