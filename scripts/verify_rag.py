#!/usr/bin/env python3
"""
Verification Script for Geology RAG
Tests hybrid semantic retrieval and statistics across all geological domains:
1. Stratigraphic Laws (Steno's Laws, Walther's Law)
2. Structural Faults & Vaults (Sibson's fault-valve, anticlinal domes)
3. Crustal Architecture (Moho, BDTZ, trans-lithospheric plumbing)
4. Ore Genesis (Hydrothermal fluids, complexation, porphyry/epithermal/orogenic)
5. Remote Sensing (ASTER/Sentinel-2 SWIR/TIR, LiDAR, aeromagnetics)
6. Ore Prospectivity ML (Spatial cross-validation, feature engineering)
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mcp_server.geology_mcp_server import query_geology_rag, get_rag_stats

console = Console()

TEST_QUERIES = [
    {
        "category": "Stratigraphic Laws & Walther's Law",
        "domain": "stratigraphic_laws",
        "query": "Walther's law facies lateral depositional environments transgressive regressive sequence stratigraphy"
    },
    {
        "category": "Steno's Principles & Chronostratigraphy",
        "domain": "stratigraphic_laws",
        "query": "Principle of superposition original horizontality cross-cutting relationships graded bedding way-up"
    },
    {
        "category": "Fault Mechanics & Structural Vaults",
        "domain": "structural_faults_vaults",
        "query": "Sibson fault-valve behavior seismic pumping dilational jogs anticlinal saddle reef structural trap"
    },
    {
        "category": "Crustal Architecture & Trans-Lithospheric Conduits",
        "domain": "crustal_architecture",
        "query": "Brittle ductile transition zone BDTZ cratonic boundary trans-lithospheric fault porphyry fertile arc Sr/Y"
    },
    {
        "category": "Ore Genesis Physicochemistry",
        "domain": "ore_genesis",
        "query": "Hydrothermal fluid gold bisulfide complexation CuCl2 chloride porphyry copper potassic phyllic alteration"
    },
    {
        "category": "Remote Sensing & Spectral Alteration",
        "domain": "remote_sensing",
        "query": "ASTER SWIR 2.2 um Al-OH absorption white mica illite kaolinite TIR quartz silicification index"
    },
    {
        "category": "Ore Prospectivity ML & Spatial Cross-Validation",
        "domain": "ore_prospectivity_ml",
        "query": "Mineral prospectivity mapping random forest spatial block cross-validation weights of evidence"
    }
]

def verify():
    console.print(Panel.fit("[bold green]Local Geology RAG - Verification Suite[/bold green]\nTesting LanceDB Hybrid Vector Retrieval & Metadata Ledgers", border_style="cyan"))

    # 1. Telemetry Stats
    stats = get_rag_stats()
    console.print(f"\n[bold]Total Ingested Papers:[/bold] {stats['total_papers']}")
    console.print(f"[bold]Total Vector Chunks Indexed:[/bold] {stats['total_chunks']}")

    if stats["breakdown"]:
        table = Table(title="Ingestion Corpus Breakdown by Domain", header_style="bold magenta")
        table.add_column("Domain", style="cyan")
        table.add_column("Subdomain", style="blue")
        table.add_column("Papers", justify="right", style="green")
        table.add_column("Indexed Chunks", justify="right", style="yellow")

        for row in stats["breakdown"]:
            table.add_row(row["domain"], row["subdomain"], str(row["papers"]), str(row["chunks"]))
        console.print(table)
    else:
        console.print("[yellow]Notice: No papers currently in the database. Run the harvester swarm to ingest literature.[/yellow]")

    # 2. Test Queries
    console.print("\n[bold cyan]─── Executing Domain Hybrid Vector Retrieval Tests ───[/bold cyan]\n")

    for idx, test in enumerate(TEST_QUERIES, 1):
        console.print(f"[bold yellow]Test {idx}: {test['category']}[/bold yellow]")
        console.print(f"[dim]Domain Filter: {test['domain']} | Query: {test['query']}[/dim]")

        results = query_geology_rag(
            query=test["query"],
            domain_filter=test["domain"],
            top_k=2
        )

        if not results:
            # Fallback to query without domain filter if empty
            results = query_geology_rag(query=test["query"], top_k=2)

        if results:
            for r_idx, hit in enumerate(results, 1):
                title = hit.get("title", "Untitled")
                year = hit.get("year", "N/A")
                domain = hit.get("domain", "")
                subdomain = hit.get("subdomain", "")
                section = hit.get("section", "Section")
                snippet = hit.get("text", "")[:350].replace("\n", " ")

                console.print(f"  [bold green]Hit {r_idx}:[/bold green] [italic]{title}[/italic] ({year}) [{domain}/{subdomain}]")
                console.print(f"  [dim]Section:[/dim] {section}")
                console.print(f"  [dim]Snippet:[/dim] {snippet}...\n")
        else:
            console.print("  [dim red]No matching chunks retrieved yet (corpus empty for this filter).[/dim red]\n")

if __name__ == "__main__":
    verify()
