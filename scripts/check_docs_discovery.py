"""Validate deployed discovery files and their targets after an HTML build."""

from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ludgerhentschel.github.io/treeig/"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.canonicals = []
        self.descriptions = []
        self.ids = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonicals.append(attrs.get("href"))
        if tag == "meta" and attrs.get("name") == "description":
            self.descriptions.append(attrs.get("content", ""))


def main():
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/_build/html"
    index = (build / "llms.txt").read_text()
    assert index == (ROOT / "llms.txt").read_text(), "Published llms.txt differs from source"
    urls = re.findall(r"\]\((https://[^\s)]+)\)", index)
    assert urls, "llms.txt has no links"
    sitemap = ET.parse(build / "sitemap.xml")
    locations = [node.text for node in sitemap.findall(".//{*}loc")]
    assert locations and len(locations) == len(set(locations)), "Empty or duplicate sitemap"
    assert all(url.startswith(BASE) for url in locations), "Incorrect sitemap base URL"
    for url in urls + locations:
        if not url.startswith(BASE):
            continue  # External targets are checked separately when publishing.
        parts = urlsplit(url[len(BASE):])
        target = build / (unquote(parts.path) or "index.html")
        assert target.is_file(), f"Missing published target: {url}"
        if target.suffix == ".html":
            page = Page(target.read_text())
            expected = BASE + target.relative_to(build).as_posix()
            assert page.canonicals == [expected], f"Incorrect canonical URL: {url}"
            if parts.fragment:
                assert unquote(parts.fragment) in page.ids, f"Missing anchor: {url}"
    for source in (ROOT / "docs").glob("*.md"):
        if "html_meta:" in source.read_text():
            page = Page((build / source.with_suffix(".html").name).read_text())
            assert len(page.descriptions) == 1 and page.descriptions[0].strip(), source
    for name in ("index", "getting-started", "models", "api", "numeric"):
        assert BASE + name + ".html" in locations, f"Guide omitted from sitemap: {name}"
    api = (build / "api.html").read_text()
    assert 'id="treeig.TreeIG"' in api, "API reference was not expanded"
    quickstart = (build / "getting-started.html").read_text()
    assert "X_train" in quickstart and "assert_allclose" in quickstart, "Quickstart was not expanded"
    print(f"Validated llms.txt, {len(locations)} sitemap URLs, canonical links, descriptions, and expanded content.")


if __name__ == "__main__":
    main()
