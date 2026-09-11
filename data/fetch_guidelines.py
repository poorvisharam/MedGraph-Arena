"""
Fetch clinical practice guidelines from HuggingFace.
Uses the epfl-llm/guidelines dataset.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GUIDELINES_DIR


def fetch_guidelines(
    output_dir: Path = GUIDELINES_DIR,
    max_docs: int = 3,
) -> list[Path]:
    """
    Fetch clinical guidelines from HuggingFace and save as .txt files.
    Returns list of saved file paths.
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    print("  [Guidelines] Loading epfl-llm/guidelines from HuggingFace...")

    try:
        dataset = load_dataset("epfl-llm/guidelines", split="train")
    except Exception as e:
        print(f"    ⚠ Could not load epfl-llm/guidelines: {e}")
        print("    → Trying alternate: falling back to manual guideline creation")
        return _create_fallback_guidelines(output_dir)

    # Take a subset focused on relevant medical topics
    keywords = ["diabetes", "hypertension", "cardiovascular", "drug", "kidney",
                 "heart", "medication", "treatment"]

    count = 0
    for i, row in enumerate(dataset):
        if count >= max_docs:
            break

        # Get text content — field names may vary
        text = row.get("text", "") or row.get("content", "") or str(row)
        title = row.get("title", "") or row.get("name", "") or f"guideline_{i}"

        # Filter for relevant content
        text_lower = text.lower()
        if not any(kw in text_lower for kw in keywords):
            continue

        # Truncate very long guidelines to keep corpus manageable
        if len(text) > 10000:
            text = text[:10000] + "\n\n[... truncated for benchmark corpus ...]"

        doc_text = (
            f"Title: {title}\n"
            f"Source: Clinical Practice Guideline (epfl-llm/guidelines)\n"
            f"Index: {i}\n"
            f"\n{text}"
        )

        safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in str(title))
        safe_name = safe_name[:80].strip().replace(" ", "_") or f"guideline_{i}"
        filepath = output_dir / f"{safe_name}.txt"
        filepath.write_text(doc_text, encoding="utf-8")
        saved_files.append(filepath)
        print(f"    ✓ Saved: {filepath.name}")
        count += 1

    # If we didn't find enough relevant guidelines, use fallback
    if len(saved_files) < max_docs:
        print(f"    → Found only {len(saved_files)} relevant guidelines, adding fallback docs")
        saved_files.extend(_create_fallback_guidelines(output_dir, max_docs - len(saved_files)))

    print(f"  [Guidelines] Total: {len(saved_files)} guidelines saved")
    return saved_files


def _create_fallback_guidelines(output_dir: Path, count: int = 3) -> list[Path]:
    """Create representative guideline documents if HuggingFace source unavailable."""
    fallback_docs = [
        {
            "title": "Type 2 Diabetes Management Guideline",
            "content": (
                "Clinical Practice Guideline: Management of Type 2 Diabetes Mellitus\n\n"
                "1. FIRST-LINE THERAPY\n"
                "Metformin remains the recommended first-line pharmacological treatment for type 2 diabetes. "
                "Starting dose: 500mg once daily, titrated to 1000mg twice daily as tolerated. "
                "Contraindicated in patients with eGFR < 30 mL/min/1.73m². Use with caution in eGFR 30-45.\n\n"
                "2. SECOND-LINE THERAPY\n"
                "If HbA1c target not achieved after 3 months of metformin monotherapy, consider adding: "
                "SGLT2 inhibitors (preferred in patients with cardiovascular disease or heart failure), "
                "GLP-1 receptor agonists (preferred in patients with atherosclerotic cardiovascular disease), "
                "DPP-4 inhibitors, sulfonylureas, or insulin.\n\n"
                "3. MONITORING\n"
                "HbA1c every 3-6 months. Annual screening for nephropathy (urine albumin-to-creatinine ratio), "
                "retinopathy, and neuropathy. Regular lipid profile and blood pressure monitoring.\n\n"
                "4. CARDIOVASCULAR RISK MANAGEMENT\n"
                "All patients with type 2 diabetes should receive cardiovascular risk assessment. "
                "Statin therapy recommended for patients aged 40-75. Aspirin therapy considered for secondary prevention. "
                "Blood pressure target < 130/80 mmHg. ACE inhibitors or ARBs preferred for hypertension with albuminuria."
            ),
        },
        {
            "title": "Hypertension Treatment Guideline",
            "content": (
                "Clinical Practice Guideline: Hypertension in Adults\n\n"
                "1. DIAGNOSIS\n"
                "Hypertension defined as sustained blood pressure ≥ 140/90 mmHg (office measurement). "
                "Confirm with ambulatory or home blood pressure monitoring. Stage 1: 140-159/90-99. Stage 2: ≥ 160/100.\n\n"
                "2. FIRST-LINE TREATMENT\n"
                "ACE inhibitors (e.g., lisinopril, ramipril) or ARBs for patients under 55 or with diabetes/CKD. "
                "Calcium channel blockers (e.g., amlodipine) or thiazide diuretics for patients over 55 or Black patients. "
                "Beta-blockers (e.g., metoprolol, bisoprolol) not first-line unless heart failure or post-MI.\n\n"
                "3. DRUG INTERACTIONS\n"
                "NSAIDs (ibuprofen, naproxen) can reduce efficacy of antihypertensives and worsen kidney function. "
                "ACE inhibitors + potassium-sparing diuretics → risk of hyperkalemia. "
                "ACE inhibitors + ARBs: combination NOT recommended (increased adverse effects without benefit).\n\n"
                "4. SPECIAL POPULATIONS\n"
                "CKD: ACE inhibitor or ARB preferred. Target BP < 130/80. Monitor potassium and creatinine. "
                "Heart failure: ACE inhibitor + beta-blocker + diuretic. Consider spironolactone. "
                "Pregnancy: Labetalol or nifedipine. ACE inhibitors and ARBs CONTRAINDICATED."
            ),
        },
        {
            "title": "Anticoagulation and Drug Interaction Guideline",
            "content": (
                "Clinical Practice Guideline: Warfarin Anticoagulation Management\n\n"
                "1. INDICATIONS\n"
                "Warfarin indicated for: atrial fibrillation (stroke prevention), deep vein thrombosis, "
                "pulmonary embolism, mechanical heart valves, and certain hypercoagulable states.\n\n"
                "2. MONITORING\n"
                "INR target 2.0-3.0 for most indications. INR 2.5-3.5 for mechanical mitral valves. "
                "Check INR at least weekly during initiation, then every 4 weeks when stable.\n\n"
                "3. CRITICAL DRUG INTERACTIONS\n"
                "INCREASED BLEEDING RISK: aspirin, NSAIDs (ibuprofen, naproxen), SSRIs, amiodarone, "
                "fluconazole, metronidazole, omeprazole (moderate effect). "
                "DECREASED EFFICACY: rifampin, carbamazepine, St. John's wort, vitamin K-rich foods.\n\n"
                "4. WARFARIN + NSAID WARNING\n"
                "Concurrent use of warfarin and NSAIDs (including ibuprofen) significantly increases "
                "risk of gastrointestinal bleeding. If analgesic needed, prefer acetaminophen. "
                "If NSAID essential, use lowest effective dose for shortest duration with PPI cover.\n\n"
                "5. REVERSAL\n"
                "For INR > 5 without bleeding: hold warfarin, consider vitamin K 1-2.5mg PO. "
                "For major bleeding: IV vitamin K 10mg + 4-factor PCC (prothrombin complex concentrate)."
            ),
        },
    ]

    saved = []
    for doc in fallback_docs[:count]:
        doc_text = (
            f"Title: {doc['title']}\n"
            f"Source: Clinical Practice Guideline\n"
            f"\n{doc['content']}"
        )
        safe_name = doc["title"].replace(" ", "_")
        filepath = output_dir / f"{safe_name}.txt"
        filepath.write_text(doc_text, encoding="utf-8")
        saved.append(filepath)
        print(f"    ✓ Saved (fallback): {filepath.name}")
    return saved


if __name__ == "__main__":
    fetch_guidelines()
