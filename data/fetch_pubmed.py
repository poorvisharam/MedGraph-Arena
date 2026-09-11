"""
Fetch PubMed abstracts using Biopython Entrez API.
Searches for drug interaction and clinical topics.
"""

import time
from pathlib import Path

from Bio import Entrez

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PUBMED_DIR, PUBMED_QUERIES, PUBMED_MAX_RESULTS_PER_QUERY


# NCBI requires an email for Entrez API usage
Entrez.email = "medgraph.arena@example.com"


def fetch_pubmed_abstracts(
    queries: list[str] = PUBMED_QUERIES,
    max_per_query: int = PUBMED_MAX_RESULTS_PER_QUERY,
    output_dir: Path = PUBMED_DIR,
) -> list[Path]:
    """
    Fetch PubMed abstracts for each query and save as .txt files.
    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for query in queries:
        print(f"  [PubMed] Searching: '{query}'...")

        # Search for PMIDs
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_per_query,
            sort="relevance",
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        pmids = search_results.get("IdList", [])

        if not pmids:
            print(f"    ⚠ No results for '{query}'")
            continue

        # Fetch full records
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="abstract",
            retmode="xml",
        )
        from Bio import Medline
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        # Parse and save each article
        articles = records.get("PubmedArticle", [])
        for article in articles:
            medline = article.get("MedlineCitation", {})
            pmid = str(medline.get("PMID", "unknown"))
            article_data = medline.get("Article", {})

            title = article_data.get("ArticleTitle", "No Title")
            abstract_parts = article_data.get("Abstract", {}).get(
                "AbstractText", []
            )
            abstract = " ".join(str(part) for part in abstract_parts)

            if not abstract:
                print(f"    ⚠ PMID {pmid}: No abstract, skipping")
                continue

            # Build document text
            doc_text = (
                f"Title: {title}\n"
                f"PMID: {pmid}\n"
                f"Source: PubMed\n"
                f"Query: {query}\n"
                f"\n{abstract}"
            )

            filepath = output_dir / f"pubmed_{pmid}.txt"
            filepath.write_text(doc_text, encoding="utf-8")
            saved_files.append(filepath)
            print(f"    ✓ Saved: {filepath.name}")

        # Be polite to NCBI API
        time.sleep(0.5)

    print(f"  [PubMed] Total: {len(saved_files)} abstracts saved")
    return saved_files


if __name__ == "__main__":
    fetch_pubmed_abstracts()
