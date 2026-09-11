"""
Fetch drug label data from openFDA API.
Retrieves drug labels for specified medications.
"""

import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FDA_DIR, FDA_DRUG_NAMES


FDA_API_URL = "https://api.fda.gov/drug/label.json"


def fetch_fda_labels(
    drug_names: list[str] = FDA_DRUG_NAMES,
    output_dir: Path = FDA_DIR,
) -> list[Path]:
    """
    Fetch drug labels from openFDA API and save as .txt files.
    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for drug in drug_names:
        print(f"  [FDA] Fetching label for: {drug}...")

        try:
            params = {
                "search": f'openfda.generic_name:"{drug}"',
                "limit": 1,
            }
            response = requests.get(FDA_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                print(f"    ⚠ No FDA label found for '{drug}'")
                continue

            label = results[0]

            # Extract key sections
            sections = {
                "indications_and_usage": label.get("indications_and_usage", []),
                "dosage_and_administration": label.get("dosage_and_administration", []),
                "warnings": label.get("warnings", []),
                "warnings_and_cautions": label.get("warnings_and_cautions", []),
                "adverse_reactions": label.get("adverse_reactions", []),
                "drug_interactions": label.get("drug_interactions", []),
                "contraindications": label.get("contraindications", []),
            }

            # Get drug metadata from openfda field
            openfda = label.get("openfda", {})
            brand_names = ", ".join(openfda.get("brand_name", ["Unknown"]))
            generic_names = ", ".join(openfda.get("generic_name", [drug]))
            manufacturer = ", ".join(openfda.get("manufacturer_name", ["Unknown"]))

            # Build document text
            doc_parts = [
                f"Drug: {generic_names}",
                f"Brand Name(s): {brand_names}",
                f"Manufacturer: {manufacturer}",
                f"Source: openFDA Drug Label",
                "",
            ]

            for section_name, content_list in sections.items():
                if content_list:
                    section_title = section_name.replace("_", " ").title()
                    content = " ".join(content_list)
                    # Truncate very long sections
                    if len(content) > 3000:
                        content = content[:3000] + " [... truncated ...]"
                    doc_parts.append(f"## {section_title}")
                    doc_parts.append(content)
                    doc_parts.append("")

            doc_text = "\n".join(doc_parts)
            filepath = output_dir / f"fda_{drug.lower()}.txt"
            filepath.write_text(doc_text, encoding="utf-8")
            saved_files.append(filepath)
            print(f"    ✓ Saved: {filepath.name}")

        except requests.exceptions.RequestException as e:
            print(f"    ⚠ Error fetching '{drug}': {e}")
        except Exception as e:
            print(f"    ⚠ Unexpected error for '{drug}': {e}")

        # Be polite to the API
        time.sleep(0.3)

    print(f"  [FDA] Total: {len(saved_files)} drug labels saved")
    return saved_files


if __name__ == "__main__":
    fetch_fda_labels()
