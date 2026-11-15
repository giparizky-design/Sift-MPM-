#!/usr/bin/env python3
"""
Autonomous Geology Literature Harvester Swarm Orchestrator
Spawns and throttles 6 specialized Subagent Harvester Instances across 16 geological topics:
1. Subagent-Stratigraphy-Sedimentology (stratigraphic_laws)
2. Subagent-Faults-Vaults-Structure (structural_faults_vaults)
3. Subagent-Crustal-Architecture (crustal_architecture)
4. Subagent-Ore-Genesis-Processes (ore_genesis)
5. Subagent-Remote-Sensing-Exploration (remote_sensing)
6. Subagent-Ore-Prospectivity-ML (ore_prospectivity_ml)

Harvests across:
- Peer-reviewed Journals (Economic Geology, J. Struct. Geol., Sedimentology, EPSL, etc.)
- Seminal Books & Treatises (Crossref, OpenAlex Books, Google Books)
- Geological Survey Bulletins & Monographs (USGS Professional Papers & Bulletins)
- Publisher APIs (Elsevier ScienceDirect, JSTOR OpenAccess & Proxy)

Enforces global asyncio.Semaphore(3) and explicit gc.collect() to guarantee
hardware-safe, low-RAM execution on Apple Silicon (16GB RAM constraint).
"""

import os
import sys
import gc
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from mcp_server.geology_mcp_server import (
    search_open_access_papers,
    search_usgs_publications,
    search_books_and_monographs,
    search_scopus_papers,
    download_and_ingest_pdf,
    get_rag_stats,
    ELSEVIER_API_KEY
)
)

console = Console()
app = typer.Typer(help="Autonomous Geology Literature Harvester Swarm Orchestrator")

TAXONOMY_PATH = BASE_DIR / "agents" / "taxonomy_manifest.json"


class SwarmTelemetry:
    def __init__(self, domains: List[str]):
        self.stats = {
            d: {
                "subagent": "",
                "status": "Idle",
                "subdomain": "-",
                "found": 0,
                "downloaded": 0,
                "chunks": 0,
                "skipped": 0,
                "errors": 0
            }
            for d in domains
        }
        self.lock = asyncio.Lock()

    async def update(self, domain: str, **kwargs):
        async with self.lock:
            if domain in self.stats:
                self.stats[domain].update(kwargs)

    def generate_table(self) -> Table:
        table = Table(
            title="🌋 Autonomous Geology Literature Harvester Swarm (Journals, Books & Surveys)",
            header_style="bold cyan",
            border_style="bright_blue"
        )
        table.add_column("Subagent Instance", style="bold white", width=36)
        table.add_column("Status", style="bold", width=12)
        table.add_column("Active Topic", style="dim cyan", width=26)
        table.add_column("Found", justify="right", style="magenta")
        table.add_column("Downloaded", justify="right", style="green")
        table.add_column("Chunks", justify="right", style="yellow")
        table.add_column("Skipped", justify="right", style="dim")
        table.add_column("Errors", justify="right", style="red")

        for domain, s in self.stats.items():
            status_style = "green" if s["status"] == "Harvesting" else ("cyan" if s["status"] == "Done" else "dim")
            table.add_row(
                s["subagent"] or domain,
                f"[{status_style}]{s['status']}[/{status_style}]",
                s["subdomain"],
                str(s["found"]),
                str(s["downloaded"]),
                str(s["chunks"]),
                str(s["skipped"]),
                str(s["errors"])
            )
        return table


async def harvest_subdomain(
    domain: str,
    subdomain: str,
    topic_info: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    telemetry: SwarmTelemetry,
    max_papers_per_query: int = 3,
    include_books: bool = True
):
    topic_name = topic_info.get("topic", subdomain)
    await telemetry.update(domain, subdomain=subdomain[:24], status="Harvesting")

    # Aggregate queries across Tier 1 (Theory), Tier 2 (Ore Genesis), Tier 3 (Remote Sensing / Geophysics)
    queries = []
    for tier in ["tier1_queries", "tier2_queries", "tier3_queries"]:
        queries.extend(topic_info.get(tier, []))

    seen_urls = set()

    for q in queries:
        # 1. Search OpenAlex & Semantic Scholar (Journals, Articles & Dissertations)
        oa_results = await search_open_access_papers(
            query=q,
            domain=domain,
            subdomain=subdomain,
            max_results=max_papers_per_query,
            include_books=include_books
        )

        # 2. Search USGS Publications Warehouse (Professional Papers, Bulletins & Models)
        usgs_results = await search_usgs_publications(
            query=q,
            domain=domain,
            subdomain=subdomain,
            max_results=max_papers_per_query
        )

        # 3. Search Books & Monographs (Crossref & OpenAlex Books)
        book_results = []
        if include_books:
            book_results = await search_books_and_monographs(
                query=q,
                domain=domain,
                subdomain=subdomain,
                max_results=max(1, max_papers_per_query // 2)
            )

        # 4. Search Elsevier Scopus API (if API key configured)
        scopus_results = []
        if ELSEVIER_API_KEY:
            scopus_results = await search_scopus_papers(
                query=q,
                domain=domain,
                subdomain=subdomain,
                max_results=max(1, max_papers_per_query // 2)
            )

        all_candidate_papers = []
        for r in oa_results + usgs_results + book_results + scopus_results:
            if "error" in r:
                continue
            pdf_url = r.get("pdf_url")
            if pdf_url and pdf_url not in seen_urls:
                seen_urls.add(pdf_url)
                all_candidate_papers.append(r)

        current_found = telemetry.stats[domain]["found"] + len(all_candidate_papers)
        await telemetry.update(domain, found=current_found)

        # Ingest discovered papers concurrently up to semaphore limit
        for paper in all_candidate_papers:
            async with semaphore:
                try:
                    res = await download_and_ingest_pdf(
                        paper_id=paper["paper_id"],
                        title=paper["title"],
                        year=paper["year"],
                        doi=paper.get("doi", ""),
                        pdf_url=paper["pdf_url"],
                        domain=domain,
                        subdomain=subdomain,
                        doc_type=paper.get("doc_type", "article")
                    )
                    status = res.get("status")
                    if status == "ingested":
                        new_dl = telemetry.stats[domain]["downloaded"] + 1
                        new_chunks = telemetry.stats[domain]["chunks"] + res.get("chunks_indexed", 0)
                        await telemetry.update(domain, downloaded=new_dl, chunks=new_chunks)
                    elif status in ("skipped_duplicate", "skipped_duplicate_hash"):
                        new_skip = telemetry.stats[domain]["skipped"] + 1
                        await telemetry.update(domain, skipped=new_skip)
                    else:
                        new_err = telemetry.stats[domain]["errors"] + 1
                        await telemetry.update(domain, errors=new_err)
                except Exception:
                    new_err = telemetry.stats[domain]["errors"] + 1
                    await telemetry.update(domain, errors=new_err)
                finally:
                    gc.collect()

                # Polite spacing between downloads
                await asyncio.sleep(0.5)


async def run_subagent_harvester(
    domain: str,
    domain_data: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    telemetry: SwarmTelemetry,
    max_subdomains: Optional[int] = None,
    max_papers_per_query: int = 3,
    include_books: bool = True
):
    subagent_name = domain_data.get("subagent_name", domain)
    subdomains = domain_data.get("subdomains", {})
    await telemetry.update(domain, subagent=subagent_name, status="Starting")

    items = list(subdomains.items())
    if max_subdomains:
        items = items[:max_subdomains]

    for subdomain, topic_info in items:
        await harvest_subdomain(
            domain=domain,
            subdomain=subdomain,
            topic_info=topic_info,
            semaphore=semaphore,
            telemetry=telemetry,
            max_papers_per_query=max_papers_per_query,
            include_books=include_books
        )

    await telemetry.update(domain, status="Done", subdomain="Complete")


async def orchestrate_swarm(
    target_domain: Optional[str] = None,
    max_subdomains: Optional[int] = None,
    max_papers_per_query: int = 3,
    concurrency_limit: int = 3,
    include_books: bool = True
):
    if not TAXONOMY_PATH.exists():
        console.print(f"[bold red]Error:[/bold red] Taxonomy manifest not found at {TAXONOMY_PATH}")
        return

    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    all_domains = taxonomy.get("domains", {})
    if target_domain:
        if target_domain not in all_domains:
            console.print(f"[bold red]Unknown domain '{target_domain}'.[/bold red] Available: {list(all_domains.keys())}")
            return
        domains_to_run = {target_domain: all_domains[target_domain]}
    else:
        domains_to_run = all_domains

    telemetry = SwarmTelemetry(list(domains_to_run.keys()))
    for d, d_val in domains_to_run.items():
        telemetry.stats[d]["subagent"] = d_val.get("subagent_name", d)

    semaphore = asyncio.Semaphore(concurrency_limit)

    console.print(Panel.fit(
        f"[bold green]Starting Comprehensive Geology Literature Swarm Orchestrator[/bold green]\n"
        f"Scope: Peer-Reviewed Journals, Seminal Books, Treatises & USGS Bulletins\n"
        f"Active Domains: {len(domains_to_run)} | Concurrency Limit: {concurrency_limit} | Max Papers/Query: {max_papers_per_query} | Include Books: {include_books}",
        border_style="cyan"
    ))

    tasks = [
        asyncio.create_task(
            run_subagent_harvester(
                domain=d,
                domain_data=d_val,
                semaphore=semaphore,
                telemetry=telemetry,
                max_subdomains=max_subdomains,
                max_papers_per_query=max_papers_per_query,
                include_books=include_books
            )
        )
        for d, d_val in domains_to_run.items()
    ]

    with Live(telemetry.generate_table(), refresh_per_second=2, console=console) as live:
        while not all(t.done() for t in tasks):
            live.update(telemetry.generate_table())
            await asyncio.sleep(0.5)
        live.update(telemetry.generate_table())

    await asyncio.gather(*tasks, return_exceptions=True)

    console.print("\n[bold green]✓ Swarm literature harvest batch complete![/bold green]")
    stats = get_rag_stats()
    console.print(f"[bold]Total Papers & Books in Corpus:[/bold] {stats['total_papers']} | [bold]Total Chunks Indexed:[/bold] {stats['total_chunks']}")


@app.command()
def start(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Specific geological domain to harvest"),
    max_subdomains: Optional[int] = typer.Option(None, "--max-subdomains", "-s", help="Max subdomains per domain to process"),
    max_papers: int = typer.Option(2, "--max-papers", "-p", help="Max papers/books discovered per query"),
    concurrency: int = typer.Option(3, "--concurrency", "-c", help="Max concurrent PDF downloads/parsers"),
    books: bool = typer.Option(True, "--include-books/--no-books", help="Search for books and treatises")
):
    """Launches the Subagent Harvester Swarm across journals, books, and papers."""
    asyncio.run(
        orchestrate_swarm(
            target_domain=domain,
            max_subdomains=max_subdomains,
            max_papers_per_query=max_papers,
            concurrency_limit=concurrency,
            include_books=books
        )
    )


if __name__ == "__main__":
    app()
