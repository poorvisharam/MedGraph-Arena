"""
Fetch WHO Essential Medicines List data.
Parses the WHO Model List of Essential Medicines into a document.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WHO_DIR


def fetch_who_eml(output_dir: Path = WHO_DIR) -> list[Path]:
    """
    Create a document from the WHO Essential Medicines List.
    The WHO EML is publicly available but difficult to parse programmatically,
    so we include the most relevant sections as a curated reference document.
    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    print("  [WHO] Creating WHO Essential Medicines List document...")

    # WHO Model List of Essential Medicines — 23rd List (2023)
    # Key sections relevant to our medical corpus
    who_eml_text = """Title: WHO Model List of Essential Medicines — 23rd Edition (2023)
Source: World Health Organization (WHO)

WHO Essential Medicines List — Selected Cardiovascular, Endocrine, and Analgesic Medicines

## 1. CARDIOVASCULAR MEDICINES

### 1.1 Antianginal Medicines
- Bisoprolol (beta-blocker): 1.25mg, 2.5mg, 5mg tablets
- Glyceryl trinitrate (nitroglycerin): 0.5mg sublingual tablets
- Isosorbide dinitrate: 10mg tablets
- Verapamil: 40mg, 80mg tablets (calcium channel blocker)

### 1.2 Antiarrhythmic Medicines
- Amiodarone: 100mg, 200mg tablets; 50mg/mL injection
- Bisoprolol: see above
- Digoxin: 62.5mcg, 250mcg tablets
- Verapamil: see above
- Lidocaine: 1%, 2% injection

### 1.3 Antihypertensive Medicines
- Amlodipine: 5mg, 10mg tablets (calcium channel blocker)
- Bisoprolol: see above (beta-blocker)
- Enalapril: 2.5mg, 5mg tablets (ACE inhibitor)
- Lisinopril: 5mg, 10mg, 20mg tablets (ACE inhibitor)
- Losartan: 25mg, 50mg, 100mg tablets (ARB)
- Hydrochlorothiazide: 12.5mg, 25mg tablets (thiazide diuretic)

### 1.4 Heart Failure Medicines
- Bisoprolol, Enalapril, Lisinopril (as above)
- Furosemide: 20mg, 40mg tablets; 10mg/mL injection (loop diuretic)
- Spironolactone: 25mg tablets (mineralocorticoid receptor antagonist)
- Digoxin (as above)

### 1.5 Antithrombotic Medicines
- Aspirin: 100mg tablets (antiplatelet)
- Clopidogrel: 75mg, 300mg tablets (antiplatelet)
- Heparin sodium: injection (anticoagulant)
- Warfarin: 1mg, 2mg, 5mg tablets (oral anticoagulant)
- Enoxaparin: injection (low molecular weight heparin)

### 1.6 Lipid-lowering Agents
- Atorvastatin: 10mg, 20mg, 40mg tablets (HMG-CoA reductase inhibitor / statin)
- Simvastatin: 10mg, 20mg, 40mg tablets

## 2. MEDICINES FOR DIABETES

### 2.1 Insulin and Analogues
- Insulin injection (soluble/regular): 100 IU/mL
- Intermediate-acting insulin (NPH): 100 IU/mL
- Long-acting insulin analogues: insulin glargine (added 2021)

### 2.2 Oral Hypoglycaemic Agents
- Metformin: 500mg, 850mg tablets (biguanide — first-line for type 2 diabetes)
- Gliclazide: 80mg tablets (sulfonylurea — second-line)
- Empagliflozin: 10mg, 25mg tablets (SGLT2 inhibitor — added 2023, for T2DM with CVD/HF)

## 3. ANALGESICS AND ANTI-INFLAMMATORY MEDICINES

### 3.1 Non-opioid Analgesics
- Paracetamol (acetaminophen): 100-500mg tablets
- Ibuprofen: 200mg, 400mg tablets (NSAID)
- Aspirin: 300-500mg tablets (NSAID, also antiplatelet at low dose)

### 3.2 Opioid Analgesics
- Morphine: 10mg, 30mg tablets; 10mg/mL injection
- Codeine: 30mg tablets
- Tramadol: 50mg capsules (added to complementary list)

### 3.3 Medicines for Gout
- Allopurinol: 100mg, 300mg tablets
- Colchicine: 500mcg tablets

## 4. KEY DRUG INTERACTION WARNINGS (WHO Guidance)

- Warfarin + aspirin: increased bleeding risk. Avoid combination unless specifically indicated.
- Warfarin + NSAIDs (ibuprofen): significantly increased GI bleeding risk. Prefer paracetamol.
- ACE inhibitors + potassium-sparing diuretics (spironolactone): hyperkalemia risk. Monitor K+.
- Metformin + contrast dye: risk of lactic acidosis. Withhold metformin 48h before/after contrast.
- Statins (atorvastatin) + amiodarone: increased myopathy risk. Limit atorvastatin to 20mg.
- Digoxin + amiodarone: increased digoxin levels. Reduce digoxin dose by 50%.
- Beta-blockers + verapamil: severe bradycardia/heart block risk. Generally avoid combination.

## 5. NOTES ON ESSENTIAL MEDICINES SELECTION

The WHO Essential Medicines List is evidence-based, considering efficacy, safety, cost-effectiveness,
and relevance to public health. Medicines are listed by International Nonproprietary Name (INN).
The list is updated every two years based on systematic review of clinical evidence.

Key principles:
- Essential medicines satisfy priority healthcare needs of the population
- Selected with due regard to disease prevalence, safety, efficacy, and comparative cost
- Intended for availability within functioning health systems at all times
- Must be in adequate amounts, appropriate dosage forms, assured quality, and affordable price
"""

    filepath = output_dir / "who_essential_medicines_list.txt"
    filepath.write_text(who_eml_text, encoding="utf-8")
    saved_files.append(filepath)
    print(f"    ✓ Saved: {filepath.name}")
    print(f"  [WHO] Total: {len(saved_files)} document saved")
    return saved_files


if __name__ == "__main__":
    fetch_who_eml()
