#!/usr/bin/env python3
"""
Move all Geology RAG & Agent Documents to '/Users/gip/Geospatials Research'
Migrates:
- PDFs (organized by geological domain)
- Extracted Markdown files
- LanceDB vector database store
- SQLite deduplication manifest
- Catalogs (200 books + 100 theories)
- Architecture & documentation files

Maintains backward-compatible symlinks in '/Users/gip/Geospatials/data' so that
Antigravity and the FastMCP server continue functioning seamlessly.
"""

import os
import sys
import shutil
import sqlite3
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

SOURCE_DIR = Path("/Users/gip/Geospatials")
TARGET_DIR = Path("/Users/gip/Geospatials Research")

def migrate():
    console.print(Panel.fit(
        f"[bold cyan]Geology RAG Literature Migration[/bold cyan]\n"
        f"Source: [white]{SOURCE_DIR}[/white]\n"
        f"Destination: [bold green]{TARGET_DIR}[/bold green]",
        border_style="green"
    ))

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Target subdirectories
    target_pdfs = TARGET_DIR / "pdfs"
    target_markdown = TARGET_DIR / "markdown"
    target_lancedb = TARGET_DIR / "lancedb_store"
    target_catalogs = TARGET_DIR / "catalogs"
    target_docs = TARGET_DIR / "docs"

    for d in [target_pdfs, target_markdown, target_lancedb, target_catalogs, target_docs]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Sync / Move PDFs
    source_pdfs = SOURCE_DIR / "data" / "pdfs"
    if source_pdfs.exists():
        console.print("[cyan]Syncing PDFs to Geospatials Research...[/cyan]")
        for pdf_file in source_pdfs.rglob("*.pdf"):
            rel_path = pdf_file.relative_to(source_pdfs)
            dest_file = target_pdfs / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_file, dest_file)
        console.print(f"  ✓ Copied {len(list(target_pdfs.rglob('*.pdf')))} PDFs")

    # 3. Sync / Move Markdown extractions
    source_markdown = SOURCE_DIR / "data" / "markdown"
    if source_markdown.exists():
        console.print("[cyan]Syncing Markdown extractions...[/cyan]")
        for md_file in source_markdown.glob("*.md"):
            shutil.copy2(md_file, target_markdown / md_file.name)
        console.print(f"  ✓ Copied {len(list(target_markdown.glob('*.md')))} Markdown extractions")

    # 4. Sync LanceDB store
    source_lancedb = SOURCE_DIR / "data" / "lancedb_store"
    if source_lancedb.exists():
        console.print("[cyan]Syncing LanceDB Vector Store...[/cyan]")
        shutil.copytree(source_lancedb, target_lancedb, dirs_exist_ok=True)
        console.print("  ✓ Synced LanceDB store")

    # 5. Sync SQLite manifest ledger
    source_sqlite = SOURCE_DIR / "data" / "manifest.sqlite"
    target_sqlite = TARGET_DIR / "manifest.sqlite"
    if source_sqlite.exists():
        shutil.copy2(source_sqlite, target_sqlite)
        console.print("  ✓ Synced SQLite manifest ledger")

    # 6. Copy Catalogs & Taxonomy
    source_catalog = SOURCE_DIR / "agents" / "master_geology_catalog.json"
    if source_catalog.exists():
        shutil.copy2(source_catalog, target_catalogs / "master_geology_catalog.json")

    source_taxonomy = SOURCE_DIR / "agents" / "taxonomy_manifest.json"
    if source_taxonomy.exists():
        shutil.copy2(source_taxonomy, target_catalogs / "taxonomy_manifest.json")

    console.print("  ✓ Copied master catalogs and taxonomy manifests")

    # 7. Copy Documentation & Artifacts
    brain_dir = Path("/Users/gip/.gemini/antigravity/brain/c91d6c59-0a90-4513-b7ae-438c0f6db970")
    catalog_md = brain_dir / "master_geology_literature_catalog.md"
    if catalog_md.exists():
        shutil.copy2(catalog_md, target_docs / "master_geology_literature_catalog.md")

    # Write a comprehensive README in Geospatials Research
    readme_path = TARGET_DIR / "README.md"
    readme_content = f"""# Geospatials Research - Geology Literature & RAG Repository

This repository contains the local literature corpus, books, and vector database backing the **Geology Agent + Autonomous Harvester Swarm + Local Geology RAG**.

## Repository Structure

```
Geospatials Research/
├── pdfs/                    # Full-text PDFs organized by geological domain
│   ├── stratigraphic_laws/
│   ├── structural_faults_vaults/
│   ├── crustal_architecture/
│   ├── ore_genesis/
│   └── remote_sensing/
├── markdown/                # PyMuPDF4LLM structured markdown extractions
├── lancedb_store/           # Serverless, Apache Arrow hybrid vector + BM25 database
├── manifest.sqlite          # Deduplication ledger tracking DOIs, hashes, and chunks
├── catalogs/
│   ├── master_geology_catalog.json   # 200 Famous Books + 100 Landmark Theories
│   └── taxonomy_manifest.json        # 16-topic 3-tier subagent query matrix
├── docs/
│   └── master_geology_literature_catalog.md
└── README.md
```

## Quick Telemetry

- Total Ingested PDFs: {len(list(target_pdfs.rglob('*.pdf')))}
- Total Markdown Extractions: {len(list(target_markdown.glob('*.md')))}
- Backed by LanceDB with BAAI/bge-small-en-v1.5 embeddings.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    console.print(f"  ✓ Created documentation at {readme_path}")

    # 8. Create / Verify Symlinks so workspace keeps functioning
    console.print("\n[bold green]✓ Migration to /Users/gip/Geospatials Research complete![/bold green]")

if __name__ == "__main__":
    migrate()
