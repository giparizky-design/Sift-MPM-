import os
import gc
import json
import sqlite3
import hashlib
import urllib.parse
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
from dotenv import load_dotenv
import pymupdf4llm
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("geology-rag-mcp")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR_ENV = os.getenv("GEOLOGY_DATA_DIR", "").strip()
if DATA_DIR_ENV and Path(DATA_DIR_ENV).exists():
    DATA_DIR = Path(DATA_DIR_ENV)
else:
    DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdfs"
MD_DIR = DATA_DIR / "markdown"
DB_DIR = DATA_DIR / "lancedb_store"
SQLITE_PATH = DATA_DIR / "manifest.sqlite"

# Credentials & Publisher Configuration from .env
ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY", "").strip()
ELSEVIER_INST_TOKEN = os.getenv("ELSEVIER_INST_TOKEN", "").strip()
JSTOR_PROXY_URL = os.getenv("JSTOR_PROXY_URL", "").strip()
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "research@local.dev").strip()

for d in [PDF_DIR, MD_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Initialize SQLite Deduplication Ledger
def init_sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            doi TEXT,
            title TEXT,
            year INTEGER,
            doc_type TEXT DEFAULT 'article',
            domain TEXT,
            subdomain TEXT,
            pdf_url TEXT,
            local_pdf_path TEXT,
            sha256 TEXT,
            chunk_count INTEGER,
            source TEXT DEFAULT 'OpenAlex',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_sqlite()

# 2. Initialize LanceDB with BGE-Small (Fast, High-Precision Scientific Embeddings)
registry = get_registry().get("sentence-transformers")
embed_model = registry.create(name="BAAI/bge-small-en-v1.5")

class GeologyChunk(LanceModel):
    chunk_id: str
    paper_id: str
    title: str
    year: int
    doi: str
    domain: str          # e.g., "stratigraphic_laws", "ore_genesis", "remote_sensing"
    subdomain: str       # e.g., "Walthers_Law", "Fault_Valve", "SWIR_Hyperspectral"
    section_header: str
    text: str = embed_model.SourceField()
    vector: Vector(embed_model.ndims()) = embed_model.VectorField()

db = lancedb.connect(str(DB_DIR))
TABLE_NAME = "geology_knowledge"

def get_or_create_table():
    if TABLE_NAME not in db.table_names():
        tbl = db.create_table(TABLE_NAME, schema=GeologyChunk)
        try:
            tbl.create_fts_index("text", replace=True)
        except Exception:
            pass
        return tbl
    return db.open_table(TABLE_NAME)

# Helper: Chunk Markdown by Geological Sections
def chunk_markdown(md_text: str, max_chars: int = 1800, overlap: int = 250) -> List[Dict[str, str]]:
    lines = md_text.split("\n")
    chunks = []
    current_header = "General / Introduction"
    buffer = []
    current_len = 0

    for line in lines:
        if line.startswith("#"):
            current_header = line.lstrip("#").strip()[:120]
        buffer.append(line)
        current_len += len(line) + 1

        if current_len >= max_chars:
            chunk_str = "\n".join(buffer).strip()
            if len(chunk_str) > 100:
                chunks.append({"header": current_header, "text": chunk_str})
            # Retain overlap
            overlap_buffer = []
            overlap_len = 0
            for prev_line in reversed(buffer):
                overlap_buffer.insert(0, prev_line)
                overlap_len += len(prev_line) + 1
                if overlap_len >= overlap:
                    break
            buffer = overlap_buffer
            current_len = overlap_len

    if buffer:
        chunk_str = "\n".join(buffer).strip()
        if len(chunk_str) > 100:
            chunks.append({"header": current_header, "text": chunk_str})
    return chunks


@mcp.tool()
async def search_open_access_papers(
    query: str,
    domain: str,
    subdomain: str,
    max_results: int = 25,
    include_books: bool = True
) -> List[Dict[str, Any]]:
    """
    Searches OpenAlex & Semantic Scholar for peer-reviewed journals, books, book chapters,
    and dissertations with direct PDF links across all publication types.
    """
    encoded_q = urllib.parse.quote(query)
    results = []
    seen_ids = set()

    # Broaden filter to include articles, books, book-chapters, dissertations, and reports
    type_filter = "type:article|book|book-chapter|report|dissertation" if include_books else "type:article"
    openalex_url = (
        f"https://api.openalex.org/works?search={encoded_q}"
        f"&filter=is_oa:true,has_fulltext:true,{type_filter}"
        f"&sort=cited_by_count:desc&per-page={min(max_results, 50)}"
    )
    headers = {"User-Agent": f"GeologyRAGHarvester/1.0 (mailto:{CROSSREF_EMAIL})"}

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(openalex_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for work in data.get("results", []):
                    best_oa = work.get("best_oa_location") or {}
                    pdf_url = best_oa.get("pdf_url") or work.get("open_access", {}).get("oa_url")
                    if not pdf_url:
                        continue

                    paper_id = work.get("id", "").split("/")[-1]
                    if paper_id in seen_ids:
                        continue
                    seen_ids.add(paper_id)

                    results.append({
                        "paper_id": paper_id,
                        "title": work.get("title") or "Untitled",
                        "year": work.get("publication_year") or 0,
                        "doi": work.get("doi") or "",
                        "doc_type": work.get("type", "article"),
                        "cited_by_count": work.get("cited_by_count", 0),
                        "pdf_url": pdf_url,
                        "source": "OpenAlex",
                        "domain": domain,
                        "subdomain": subdomain
                    })
    except Exception as e:
        results.append({"error": f"OpenAlex query error: {str(e)}"})

    # 2. Query Semantic Scholar
    if len(results) < max_results:
        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_q}&limit={min(max_results, 20)}&fields=title,year,openAccessPdf,citationCount,externalIds,publicationTypes"
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                s2_resp = await client.get(s2_url, headers=headers)
                if s2_resp.status_code == 200:
                    s2_data = s2_resp.json()
                    for item in s2_data.get("data", []):
                        oa_info = item.get("openAccessPdf") or {}
                        pdf_url = oa_info.get("url")
                        if not pdf_url:
                            continue
                        paper_id = item.get("paperId", "")
                        if paper_id in seen_ids:
                            continue
                        seen_ids.add(paper_id)
                        doi = (item.get("externalIds") or {}).get("DOI", "")
                        pub_types = item.get("publicationTypes") or ["JournalArticle"]

                        results.append({
                            "paper_id": paper_id,
                            "title": item.get("title") or "Untitled",
                            "year": item.get("year") or 0,
                            "doi": doi,
                            "doc_type": pub_types[0] if pub_types else "article",
                            "cited_by_count": item.get("citationCount", 0),
                            "pdf_url": pdf_url,
                            "source": "SemanticScholar",
                            "domain": domain,
                            "subdomain": subdomain
                        })
        except Exception:
            pass

    return results[:max_results]


@mcp.tool()
async def search_usgs_publications(
    query: str,
    domain: str,
    subdomain: str,
    max_results: int = 25
) -> List[Dict[str, Any]]:
    """
    Searches the USGS Publications Warehouse for mineral deposit models, prospectivity,
    bulletins, professional papers, and spectral manuals.
    """
    encoded_q = urllib.parse.quote(query)
    url = f"https://pubs.er.usgs.gov/pubs-services/publication?q={encoded_q}&page_size={min(max_results, 50)}"
    headers = {
        "Accept": "application/json",
        "User-Agent": f"GeologyRAGHarvester/1.0 (mailto:{CROSSREF_EMAIL})"
    }
    results = []

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return [{"error": f"USGS API HTTP {resp.status_code}"}]
            data = resp.json()

            for record in data.get("records", []):
                pdf_url = None
                links = record.get("links", [])
                for link in links:
                    u = link.get("url", "")
                    link_type = link.get("type", {})
                    link_text = link_type.get("text", "") if isinstance(link_type, dict) else str(link_type)
                    if u.lower().endswith(".pdf") or link_text.lower() in ("document", "report"):
                        if u.startswith("http") and ".pdf" in u.lower():
                            pdf_url = u
                            break

                if not pdf_url:
                    continue

                index_id = record.get("indexId") or record.get("id") or str(record.get("publicationId", ""))
                year_val = record.get("publicationYear") or 0
                try:
                    year_val = int(year_val)
                except (ValueError, TypeError):
                    year_val = 0

                pub_type = record.get("publicationSubtype", {}).get("text", "USGS Publication") if isinstance(record.get("publicationSubtype"), dict) else "USGS Publication"

                results.append({
                    "paper_id": f"usgs_{index_id}",
                    "title": record.get("title") or "Untitled USGS Publication",
                    "year": year_val,
                    "doi": record.get("doi") or "",
                    "doc_type": pub_type,
                    "cited_by_count": 0,
                    "pdf_url": pdf_url,
                    "source": "USGS",
                    "domain": domain,
                    "subdomain": subdomain
                })
    except Exception as e:
        return [{"error": f"USGS query error: {str(e)}"}]

    return results[:max_results]


@mcp.tool()
async def search_books_and_monographs(
    query: str,
    domain: str,
    subdomain: str,
    max_results: int = 15
) -> List[Dict[str, Any]]:
    """
    Searches Crossref and OpenAlex specifically for books, monographs, and book chapters.
    """
    encoded_q = urllib.parse.quote(query)
    results = []
    seen = set()

    # 1. Crossref Books & Monographs
    crossref_url = (
        f"https://api.crossref.org/works?query={encoded_q}"
        f"&filter=type:book,type:monograph&rows={min(max_results, 25)}"
    )
    headers = {"User-Agent": f"GeologyRAGHarvester/1.0 (mailto:{CROSSREF_EMAIL})"}

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(crossref_url, headers=headers)
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items", [])
                for item in items:
                    title_list = item.get("title", [])
                    title = title_list[0] if title_list else "Untitled Book"
                    doi = item.get("DOI", "")
                    if doi in seen:
                        continue
                    seen.add(doi)

                    year = 0
                    published = item.get("published-print") or item.get("published-online") or {}
                    date_parts = published.get("date-parts", [[]])[0]
                    if date_parts:
                        year = date_parts[0]

                    publisher = item.get("publisher", "Unknown Publisher")

                    results.append({
                        "paper_id": f"book_{doi.replace('/', '_')}",
                        "title": title,
                        "year": year,
                        "doi": doi,
                        "doc_type": "book",
                        "publisher": publisher,
                        "source": "CrossrefBooks",
                        "domain": domain,
                        "subdomain": subdomain
                    })
    except Exception:
        pass

    # 2. OpenAlex Books with Open Access PDFs
    oa_books_url = (
        f"https://api.openalex.org/works?search={encoded_q}"
        f"&filter=is_oa:true,type:book|book-chapter"
        f"&per-page={min(max_results, 20)}"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            oa_resp = await client.get(oa_books_url, headers=headers)
            if oa_resp.status_code == 200:
                for work in oa_resp.json().get("results", []):
                    best_oa = work.get("best_oa_location") or {}
                    pdf_url = best_oa.get("pdf_url")
                    if not pdf_url:
                        continue
                    paper_id = work.get("id", "").split("/")[-1]
                    if paper_id in seen:
                        continue
                    seen.add(paper_id)

                    results.append({
                        "paper_id": paper_id,
                        "title": work.get("title") or "Untitled Book",
                        "year": work.get("publication_year") or 0,
                        "doi": work.get("doi") or "",
                        "doc_type": work.get("type", "book"),
                        "pdf_url": pdf_url,
                        "source": "OpenAlexBooks",
                        "domain": domain,
                        "subdomain": subdomain
                    })
    except Exception:
        pass

    return results[:max_results]


@mcp.tool()
async def search_scopus_papers(
    query: str,
    domain: str,
    subdomain: str,
    max_results: int = 15
) -> List[Dict[str, Any]]:
    """
    Searches Elsevier Scopus API across 25,000+ peer-reviewed geology journals, books, and proceedings
    using the configured ELSEVIER_API_KEY, and resolves DOIs to open-access PDF links via OpenAlex.
    """
    if not ELSEVIER_API_KEY:
        return [{"error": "ELSEVIER_API_KEY not configured in .env"}]

    clean_q = query.replace('"', '').replace("'", "")
    scopus_url = f"https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY({urllib.parse.quote(clean_q)})&count={min(max_results, 25)}"
    headers = {
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept": "application/json"
    }

    results = []
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            resp = await client.get(scopus_url, headers=headers)
            if resp.status_code != 200:
                return [{"error": f"Scopus API HTTP {resp.status_code}", "detail": resp.text[:200]}]

            entries = resp.json().get("search-results", {}).get("entry", [])
            for e in entries:
                title = e.get("dc:title") or "Untitled Scopus Record"
                doi = e.get("prism:doi") or ""
                cover_date = e.get("prism:coverDate") or ""
                year_val = 0
                if cover_date and len(cover_date) >= 4:
                    try:
                        year_val = int(cover_date[:4])
                    except ValueError:
                        year_val = 0

                pub_name = e.get("prism:publicationName") or "Elsevier/Scopus Journal"
                agg_type = e.get("prism:aggregationType") or "Journal"
                oa_flag = str(e.get("openaccessFlag", "false")).lower() == "true" or e.get("openaccess") == "1"

                pdf_url = None
                # If DOI exists, attempt quick OpenAlex OA PDF lookup
                if doi:
                    try:
                        oa_lookup = await client.get(
                            f"https://api.openalex.org/works/https://doi.org/{doi}",
                            headers={"User-Agent": f"GeologyHarvester/1.0 (mailto:{CROSSREF_EMAIL})"},
                            timeout=8.0
                        )
                        if oa_lookup.status_code == 200:
                            work_data = oa_lookup.json()
                            best_oa = work_data.get("best_oa_location") or {}
                            pdf_url = best_oa.get("pdf_url") or work_data.get("open_access", {}).get("oa_url")
                    except Exception:
                        pass

                scopus_id = e.get("dc:identifier", "").replace("SCOPUS_ID:", "") or doi.replace("/", "_")
                results.append({
                    "paper_id": f"scopus_{scopus_id}",
                    "title": title,
                    "year": year_val,
                    "doi": doi,
                    "journal": pub_name,
                    "doc_type": agg_type.lower(),
                    "is_oa": oa_flag,
                    "pdf_url": pdf_url,
                    "source": "ElsevierScopus",
                    "domain": domain,
                    "subdomain": subdomain
                })
    except Exception as exc:
        return [{"error": f"Scopus query error: {str(exc)}"}]

    return results[:max_results]


@mcp.tool()
async def download_and_ingest_pdf(
    paper_id: str,
    title: str,
    year: int,
    doi: str,
    pdf_url: str,
    domain: str,
    subdomain: str,
    doc_type: str = "article"
) -> Dict[str, Any]:
    """
    Downloads a journal paper or book PDF (via OpenAlex, USGS, Elsevier API, or JSTOR proxy),
    verifies SHA256 deduplication, converts to Markdown, chunks, embeds with BGE-Small,
    and indexes into the local LanceDB Geology RAG.
    """
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("SELECT paper_id FROM papers WHERE paper_id = ? OR (doi != '' AND doi = ?)", (paper_id, doi))
    if c.fetchone():
        conn.close()
        return {"status": "skipped_duplicate", "paper_id": paper_id, "title": title}

    domain_dir = PDF_DIR / domain / subdomain
    domain_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch for ch in paper_id if ch.isalnum() or ch in ("_", "-"))
    pdf_path = domain_dir / f"{safe_id}.pdf"

    # Assemble request headers with publisher API keys if available
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/xhtml+xml,*/*"
    }

    # Elsevier ScienceDirect API Integration
    target_url = pdf_url
    if ("elsevier.com" in pdf_url.lower() or "sciencedirect.com" in pdf_url.lower()) and ELSEVIER_API_KEY:
        headers["X-ELS-APIKey"] = ELSEVIER_API_KEY
        if ELSEVIER_INST_TOKEN:
            headers["X-ELS-Insttoken"] = ELSEVIER_INST_TOKEN
        headers["Accept"] = "application/pdf"
        if doi:
            clean_doi = doi.replace("https://doi.org/", "")
            target_url = f"https://api.elsevier.com/content/article/doi/{clean_doi}"

    # JSTOR Institutional Proxy Integration
    elif "jstor.org" in pdf_url.lower() and JSTOR_PROXY_URL:
        target_url = f"{JSTOR_PROXY_URL.rstrip('/')}/{pdf_url}"

    try:
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            async with client.stream("GET", target_url, headers=headers) as resp:
                if resp.status_code != 200:
                    conn.close()
                    return {"status": "invalid_pdf_or_paywall", "http_status": resp.status_code, "url": target_url}

                hasher = hashlib.sha256()
                first_chunk = True
                with open(pdf_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        if first_chunk:
                            first_chunk = False
                            if b"%PDF" not in chunk[:2048]:
                                f.close()
                                pdf_path.unlink(missing_ok=True)
                                conn.close()
                                return {"status": "invalid_pdf_or_paywall", "http_status": resp.status_code, "url": target_url}
                        hasher.update(chunk)
                        f.write(chunk)

            sha256_hash = hasher.hexdigest()
            c.execute("SELECT paper_id FROM papers WHERE sha256 = ?", (sha256_hash,))
            if c.fetchone():
                pdf_path.unlink(missing_ok=True)
                conn.close()
                return {"status": "skipped_duplicate_hash", "paper_id": paper_id}

        # Convert PDF to clean Markdown
        # For large documents (>60 pages), use fast TOC-aware extraction (<2s) to avoid heavy layout timeouts
        doc = pymupdf.open(str(pdf_path))
        page_count = len(doc)
        if page_count > 60:
            toc = doc.get_toc()
            toc_map = {}
            for item in toc:
                lvl, t, pno = item[0], item[1], item[2]
                toc_map.setdefault(pno, []).append((lvl, t))

            pages_out = []
            for p_idx in range(page_count):
                pno = p_idx + 1
                if pno in toc_map:
                    for lvl, t in toc_map[pno]:
                        prefix = '#' * min(lvl + 1, 4)
                        pages_out.append(f'\n{prefix} {t}\n')
                try:
                    page = doc.load_page(p_idx)
                    txt = page.get_text()
                    if txt.strip():
                        pages_out.append(txt.strip())
                except Exception:
                    continue
            doc.close()
            md_text = '\n\n'.join(pages_out)
        else:
            doc.close()
            try:
                pymupdf4llm.use_layout(False)
                md_text = pymupdf4llm.to_markdown(str(pdf_path))
            except Exception:
                try:
                    d_fb = pymupdf.open(str(pdf_path))
                    fb_pages = []
                    for p_idx in range(len(d_fb)):
                        try:
                            fb_pages.append(d_fb.load_page(p_idx).get_text())
                        except Exception:
                            continue
                    d_fb.close()
                    md_text = '\n\n'.join(fb_pages)
                except Exception:
                    md_text = ""

        if len(md_text.strip()) < 500:
            pdf_path.unlink(missing_ok=True)
            conn.close()
            return {"status": "empty_or_scanned_pdf", "paper_id": paper_id}

        md_file = MD_DIR / f"{safe_id}.md"
        md_file.write_text(md_text, encoding="utf-8")

        raw_chunks = chunk_markdown(md_text)
        records = []
        for idx, ch in enumerate(raw_chunks):
            records.append({
                "chunk_id": f"{safe_id}_{idx}",
                "paper_id": safe_id,
                "title": title,
                "year": int(year) if year else 0,
                "doi": doi or "",
                "domain": domain,
                "subdomain": subdomain,
                "section_header": ch["header"],
                "text": f"[{title} ({year}) | {doc_type} | {domain}/{subdomain} | {ch['header']}]\n{ch['text']}"
            })

        tbl = get_or_create_table()
        # Batch insert to keep memory usage low (<4GB RAM on Mac)
        batch_size = 64
        for i in range(0, len(records), batch_size):
            tbl.add(records[i:i + batch_size])

        c.execute(
            "INSERT INTO papers (paper_id, doi, title, year, doc_type, domain, subdomain, pdf_url, local_pdf_path, sha256, chunk_count, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (safe_id, doi, title, year, doc_type, domain, subdomain, pdf_url, str(pdf_path), sha256_hash, len(records), "Publisher/Harvester")
        )
        conn.commit()
        conn.close()
        gc.collect()

        return {
            "status": "ingested",
            "paper_id": safe_id,
            "title": title,
            "doc_type": doc_type,
            "chunks_indexed": len(records)
        }
    except Exception as e:
        conn.close()
        return {"status": "error", "paper_id": paper_id, "error": str(e)}


@mcp.tool()
def query_geology_rag(
    query: str,
    domain_filter: Optional[str] = None,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Queries the local LanceDB Geology RAG using semantic vector search + optional domain filter.
    """
    tbl = get_or_create_table()
    search_builder = tbl.search(query).limit(top_k)
    if domain_filter:
        search_builder = search_builder.where(f"domain = '{domain_filter}'")
    results = search_builder.to_list()
    return [
        {
            "title": r.get("title", ""),
            "year": r.get("year", 0),
            "doi": r.get("doi", ""),
            "domain": r.get("domain", ""),
            "subdomain": r.get("subdomain", ""),
            "section": r.get("section_header", ""),
            "text": r.get("text", "")
        }
        for r in results
    ]


@mcp.tool()
def get_rag_stats() -> Dict[str, Any]:
    """Returns ingestion telemetry across all geological domains, subdomains, and document types."""
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("SELECT domain, subdomain, COUNT(*), SUM(chunk_count) FROM papers GROUP BY domain, subdomain")
    rows = c.fetchall()
    conn.close()
    breakdown = [
        {"domain": r[0], "subdomain": r[1], "papers": r[2], "chunks": r[3] or 0}
        for r in rows
    ]
    return {
        "total_papers": sum(r["papers"] for r in breakdown),
        "total_chunks": sum(r["chunks"] for r in breakdown if r["chunks"]),
        "breakdown": breakdown
    }

if __name__ == "__main__":
    mcp.run()
