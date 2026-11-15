#!/usr/bin/env python3
"""
Famous Geology Literature & Theory Harvester Orchestrator
Targeted harvester for:
- 200 Landmark Geology Textbooks & Treatises
- 100 Landmark Geological Theories & Seminal Papers

Main Outlets:
1. ScienceDirect / Elsevier Scopus (via ELSEVIER_API_KEY)
2. Library Genesis (LibGen) & Anna's Archive (via mirror resolvers)
3. Internet Archive (Archive.org Texts API for digitized classic books)
4. OpenAlex & Crossref (for verified OA monographs and state survey publications)
5. USGS Publications Warehouse (for Professional Papers and Bulletin models)

Hardware-safe: Enforces asyncio.Semaphore(3) and explicit gc.collect() (<4GB RAM on Mac).
"""

import os
import sys
import gc
import json
import asyncio
import re
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx
import typer
import sqlite3
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from mcp_server.geology_mcp_server import (
    download_and_ingest_pdf,
    search_scopus_papers,
    get_rag_stats,
    ELSEVIER_API_KEY,
    CROSSREF_EMAIL,
    SQLITE_PATH
)

console = Console()
app = typer.Typer(help="Famous Geology Literature & Theory Harvester Orchestrator")

CATALOG_PATH = BASE_DIR / "agents" / "master_geology_catalog.json"


class LiteratureTelemetry:
    def __init__(self):
        self.stats = {
            "Books": {"total": 0, "found": 0, "downloaded": 0, "chunks": 0, "skipped": 0, "failed": 0},
            "Theories": {"total": 0, "found": 0, "downloaded": 0, "chunks": 0, "skipped": 0, "failed": 0}
        }
        self.sources = {
            "ScienceDirect/Scopus": 0,
            "LibGen/Anna": 0,
            "InternetArchive": 0,
            "OpenAlex/Crossref": 0,
            "USGS": 0
        }
        self.current_item = "Initializing..."
        self.lock = asyncio.Lock()

    async def update_item(self, item_name: str):
        async with self.lock:
            self.current_item = item_name[:45]

    async def record_event(self, item_type: str, event: str, count: int = 1, source: Optional[str] = None):
        async with self.lock:
            if item_type in self.stats and event in self.stats[item_type]:
                self.stats[item_type][event] += count
            if source and source in self.sources:
                self.sources[source] += count

    def generate_table(self) -> Table:
        table = Table(
            title="📚 Landmark Geology Literature & Theory Harvester - Live Swarm",
            header_style="bold cyan",
            border_style="bright_blue"
        )
        table.add_column("Category", style="bold white", width=14)
        table.add_column("Total Catalog", justify="right", style="cyan", width=14)
        table.add_column("Discovered", justify="right", style="magenta", width=12)
        table.add_column("Downloaded", justify="right", style="green", width=12)
        table.add_column("Chunks", justify="right", style="yellow", width=10)
        table.add_column("Skipped", justify="right", style="dim", width=10)
        table.add_column("Failed", justify="right", style="red", width=10)

        for cat, s in self.stats.items():
            table.add_row(
                cat,
                str(s["total"]),
                str(s["found"]),
                str(s["downloaded"]),
                str(s["chunks"]),
                str(s["skipped"]),
                str(s["failed"])
            )
        return table


GRAMMAR_STOP_WORDS = {
    'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'on', 'with', 'by', 'from',
    'at', 'into', 'under', 'as', 'is', 'its', 'their', 'vol', 'volume', 'part'
}


def clean_search_keywords(title: str, max_words: int = 4) -> str:
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', title).strip()
    words = [w for w in clean.split() if w.lower() not in GRAMMAR_STOP_WORDS]
    return '+'.join(words[:max_words])


def extract_primary_author_surname(author_str: str) -> str:
    if not author_str or author_str.lower() == 'unknown':
        return ''
    clean = re.sub(r'\bet al\.?', '', author_str, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in re.split(r'[,;]', clean) if p.strip()]
    if not parts:
        return ''
    tokens = [t for t in re.sub(r'[^a-zA-Z\s]', '', parts[0]).split() if t.lower() not in {'jr', 'sr', 'ii', 'iii', 'dr'}]
    return tokens[-1] if tokens else ''


async def resolve_sciencedirect_scopus(title: str, author: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Resolves item via Elsevier Scopus & ScienceDirect API."""
    if not ELSEVIER_API_KEY:
        return None

    surname = extract_primary_author_surname(author)
    auth_clause = f" AND AUTH({surname})" if surname else ""
    clean_q = f"TITLE({title}){auth_clause}"
    scopus_url = f"https://api.elsevier.com/content/search/scopus?query={urllib.parse.quote(clean_q)}&count=2"
    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}

    try:
        resp = await client.get(scopus_url, headers=headers, timeout=12.0)
        if resp.status_code == 200:
            entries = resp.json().get("search-results", {}).get("entry", [])
            for e in entries:
                doi = e.get("prism:doi")
                if not doi:
                    continue
                pdf_lookup = f"https://api.openalex.org/works/https://doi.org/{doi}"
                oa_resp = await client.get(pdf_lookup, headers={"User-Agent": f"GeologyHarvester/1.0 (mailto:{CROSSREF_EMAIL})"}, timeout=8.0)
                if oa_resp.status_code == 200:
                    best_oa = oa_resp.json().get("best_oa_location") or {}
                    pdf_url = best_oa.get("pdf_url")
                    if pdf_url:
                        return {
                            "title": e.get("dc:title", title),
                            "doi": doi,
                            "pdf_url": pdf_url,
                            "source": "ScienceDirect/Scopus"
                        }
    except Exception:
        pass
    return None


async def resolve_libgen(title: str, author: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Resolves item via LibGen mirror endpoints with author ranking and get.php extraction."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    surname = extract_primary_author_surname(author)

    query_attempts = [
        clean_search_keywords(title, 4),
        f"{clean_search_keywords(title, 3)}+{surname}" if surname else clean_search_keywords(title, 3),
        clean_search_keywords(title, 2)
    ]

    for q in query_attempts:
        if not q:
            continue
        libgen_url = f"https://libgen.li/index.php?req={q}&columns%5B%5D=t"
        try:
            resp = await client.get(libgen_url, headers=headers, timeout=12.0)
            if resp.status_code != 200:
                continue

            rows = re.findall(r'<tr[^>]*>.*?</tr>', resp.text, flags=re.DOTALL)
            best_row = None
            best_score = -1
            for row in rows:
                if "ads.php" not in row:
                    continue
                clean = re.sub(r'<[^>]+>', ' ', row).strip()
                clean_lower = clean.lower()
                # Skip journal reviews and errata to ensure full textbook retrieval
                if "book review" in clean_lower or "erratum" in clean_lower:
                    continue

                score = 0
                if surname and surname.lower() in clean_lower:
                    score += 25
                if " mb" in clean_lower:
                    score += 20
                elif " kb" in clean_lower:
                    score -= 20
                if "pdf" in clean_lower:
                    score += 10
                if "english" in clean_lower:
                    score += 2

                if score > best_score:
                    best_score = score
                    best_row = row

            if best_row:
                matches = re.findall(r'href=[\"\'](/??ads\.php\?md5=[a-zA-Z0-9]+[^\"\']*)[\"\']', best_row)
                if matches:
                    ads_url = urllib.parse.urljoin("https://libgen.li/", matches[0])
                    r_ads = await client.get(ads_url, headers=headers, timeout=12.0)
                    if r_ads.status_code == 200:
                        get_matches = re.findall(r'href=[\"\'](/??get\.php\?[^\"\']+)[\"\']', r_ads.text)
                        if get_matches:
                            dl_url = urllib.parse.urljoin("https://libgen.li/", get_matches[0])
                            return {
                                "title": title,
                                "doi": "",
                                "pdf_url": dl_url,
                                "source": "LibGen/Anna"
                            }
        except Exception:
            pass
    return None


async def resolve_usgs(title: str, author: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Resolves USGS Bulletins, Professional Papers, and Open-File Reports."""
    encoded_q = urllib.parse.quote(title)
    usgs_url = f"https://pubs.er.usgs.gov/pubs-services/publication?q={encoded_q}&page_size=4"
    headers = {"Accept": "application/json", "User-Agent": f"GeologyHarvester/1.0 (mailto:{CROSSREF_EMAIL})"}

    try:
        resp = await client.get(usgs_url, headers=headers, timeout=10.0)
        if resp.status_code == 200:
            for rec in resp.json().get("records", []):
                for link in rec.get("links", []):
                    u = link.get("url", "")
                    if u.lower().endswith(".pdf") and u.startswith("http"):
                        return {
                            "title": rec.get("title", title),
                            "doi": rec.get("doi", ""),
                            "pdf_url": u,
                            "source": "USGS"
                        }
    except Exception:
        pass
    return None


async def resolve_internet_archive(title: str, author: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Resolves classic geology textbooks & physical geology treatises via Internet Archive."""
    words = clean_search_keywords(title, 3)
    surname = extract_primary_author_surname(author)
    auth_q = f"+AND+creator%3A%28{surname}%29" if surname else ""
    ia_url = f"https://archive.org/advancedsearch.php?q=title%3A%28{words}%29{auth_q}+AND+mediatype%3Atexts&fl[]=identifier,title,creator,year&rows=2&output=json"

    try:
        resp = await client.get(ia_url, timeout=10.0)
        if resp.status_code == 200:
            docs = resp.json().get("response", {}).get("docs", [])
            for d in docs:
                ident = d.get("identifier")
                if ident:
                    pdf_url = f"https://archive.org/download/{ident}/{ident}.pdf"
                    return {
                        "title": d.get("title", title),
                        "doi": "",
                        "pdf_url": pdf_url,
                        "source": "InternetArchive"
                    }
    except Exception:
        pass
    return None


async def resolve_openalex_usgs(title: str, author: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Fallback resolver via OpenAlex works API."""
    encoded_q = urllib.parse.quote(f"{title} {extract_primary_author_surname(author)}")
    oa_url = f"https://api.openalex.org/works?search={encoded_q}&filter=is_oa:true,has_fulltext:true&per-page=2"
    headers = {"User-Agent": f"GeologyHarvester/1.0 (mailto:{CROSSREF_EMAIL})"}

    try:
        resp = await client.get(oa_url, headers=headers, timeout=10.0)
        if resp.status_code == 200:
            for w in resp.json().get("results", []):
                best_oa = w.get("best_oa_location") or {}
                pdf_url = best_oa.get("pdf_url") or w.get("open_access", {}).get("oa_url")
                if pdf_url:
                    return {
                        "title": w.get("title", title),
                        "doi": w.get("doi", ""),
                        "pdf_url": pdf_url,
                        "source": "OpenAlex/Crossref"
                    }
    except Exception:
        pass
    return None


async def harvest_single_entry(
    entry: Dict[str, Any],
    item_type: str,
    semaphore: asyncio.Semaphore,
    telemetry: LiteratureTelemetry,
    client: httpx.AsyncClient
):
    title = entry.get("title") or entry.get("name", "Untitled")
    author = entry.get("authors") or entry.get("author", "Unknown")
    year = entry.get("year", 0)
    domain = entry.get("domain", "stratigraphic_laws")
    subdomain = entry.get("subdomain", "General")
    doc_id = entry.get("id", "doc")

    # Fast check: skip if already in manifest.sqlite
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        c = conn.cursor()
        c.execute("SELECT paper_id FROM papers WHERE paper_id = ? OR LOWER(title) = LOWER(?)", (doc_id, title))
        if c.fetchone():
            conn.close()
            await telemetry.record_event(item_type, "skipped", 1)
            return
        conn.close()
    except Exception:
        pass

    await telemetry.update_item(f"[{item_type}] {title}")

    # Pipeline resolution order: USGS-specific first for survey publications, else LibGen/Anna first
    if "usgs" in title.lower() or "bulletin" in title.lower() or "survey" in title.lower():
        resolvers = [resolve_usgs, resolve_libgen, resolve_internet_archive, resolve_sciencedirect_scopus, resolve_openalex_usgs]
    else:
        resolvers = [resolve_libgen, resolve_internet_archive, resolve_usgs, resolve_sciencedirect_scopus, resolve_openalex_usgs]

    resolved = None
    for resolver in resolvers:
        resolved = await resolver(title, author, client)
        if resolved:
            break

    if not resolved:
        await telemetry.record_event(item_type, "failed", 1)
        return

    await telemetry.record_event(item_type, "found", 1)
    source_name = resolved.get("source", "Unknown")

    async with semaphore:
        try:
            res = await download_and_ingest_pdf(
                paper_id=doc_id,
                title=resolved.get("title", title),
                year=year,
                doi=resolved.get("doi", ""),
                pdf_url=resolved["pdf_url"],
                domain=domain,
                subdomain=subdomain,
                doc_type="book" if item_type == "Books" else "seminal_theory"
            )

            status = res.get("status")
            if status == "ingested":
                chunks = res.get("chunks_indexed", 0)
                await telemetry.record_event(item_type, "downloaded", 1, source=source_name)
                await telemetry.record_event(item_type, "chunks", chunks)
            elif status in ("skipped_duplicate", "skipped_duplicate_hash"):
                await telemetry.record_event(item_type, "skipped", 1)
            else:
                await telemetry.record_event(item_type, "failed", 1)
        except Exception:
            await telemetry.record_event(item_type, "failed", 1)
        finally:
            gc.collect()

        await asyncio.sleep(0.5)


async def run_master_harvester(target: str = "all", limit: Optional[int] = None, concurrency: int = 3):
    if not CATALOG_PATH.exists():
        console.print(f"[bold red]Catalog not found at {CATALOG_PATH}. Run master_geology_catalog.py first.[/bold red]")
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    books = catalog.get("books", [])
    theories = catalog.get("theories", [])

    entries_to_harvest = []
    if target in ("all", "books"):
        for b in (books[:limit] if limit else books):
            entries_to_harvest.append((b, "Books"))
    if target in ("all", "theories"):
        for t in (theories[:limit] if limit else theories):
            entries_to_harvest.append((t, "Theories"))

    telemetry = LiteratureTelemetry()
    telemetry.stats["Books"]["total"] = len(books)
    telemetry.stats["Theories"]["total"] = len(theories)

    semaphore = asyncio.Semaphore(concurrency)

    console.print(Panel.fit(
        f"[bold green]Starting Famous Geology Literature & Theory Swarm Orchestrator[/bold green]\n"
        f"Target Outlets: [bold cyan]ScienceDirect (Scopus) & LibGen / Internet Archive[/bold cyan]\n"
        f"Queued Items: {len(entries_to_harvest)} | Concurrency Semaphore: {concurrency} | Memory Guard: <4GB RAM",
        border_style="cyan"
    ))

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        queue = asyncio.Queue()
        for item in entries_to_harvest:
            queue.put_nowait(item)

        async def worker(worker_id: int):
            while not queue.empty():
                try:
                    entry, item_type = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await harvest_single_entry(entry, item_type, semaphore, telemetry, client)
                except Exception:
                    pass
                finally:
                    queue.task_done()
                # Polite rate-limit spacing between searches (protects Scopus & LibGen quotas)
                await asyncio.sleep(1.0)

        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]

        with Live(telemetry.generate_table(), refresh_per_second=2, console=console) as live:
            while not all(w.done() for w in workers):
                live.update(telemetry.generate_table())
                await asyncio.sleep(0.5)
            live.update(telemetry.generate_table())

        await asyncio.gather(*workers, return_exceptions=True)

    console.print("\n[bold green]✓ Master literature & theory harvest run complete![/bold green]")
    stats = get_rag_stats()
    console.print(f"[bold]Total Registered Documents in RAG:[/bold] {stats['total_papers']}")
    console.print(f"[bold]Total Vector Chunks Indexed:[/bold] {stats['total_chunks']}")


@app.command()
def start(
    target: str = typer.Option("all", "--target", "-t", help="Harvest target: 'all', 'books', or 'theories'"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of items to harvest (default: all)"),
    concurrency: int = typer.Option(3, "--concurrency", "-c", help="Concurrent downloads (default: 3)")
):
    """Launches the Famous Geology Books & Theories Harvester Swarm."""
    asyncio.run(run_master_harvester(target=target, limit=limit, concurrency=concurrency))


if __name__ == "__main__":
    app()
