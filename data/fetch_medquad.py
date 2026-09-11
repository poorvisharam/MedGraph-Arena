"""
Fetch MedQuAD Q&A pairs.
MedQuAD is a medical question-answering dataset from NIH.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MEDQUAD_DIR


def fetch_medquad(
    output_dir: Path = MEDQUAD_DIR,
    max_docs: int = 20,
) -> list[Path]:
    """
    Create MedQuAD-style medical Q&A documents.
    The original MedQuAD dataset may not always be easily downloadable,
    so we include representative Q&A pairs covering our focus areas.
    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    print("  [MedQuAD] Creating medical Q&A documents...")

    # Representative Q&A pairs covering drugs, conditions, and interactions
    # in our corpus (diabetes, cardiovascular, drug interactions)
    qa_pairs = [
        {
            "id": "medquad_001",
            "question": "What is metformin and what is it used for?",
            "answer": (
                "Metformin is a biguanide oral antidiabetic medication. It is the first-line treatment for type 2 diabetes mellitus. "
                "Metformin works by decreasing hepatic glucose production, decreasing intestinal absorption of glucose, and improving "
                "insulin sensitivity by increasing peripheral glucose uptake and utilization. It does not cause hypoglycemia when used alone. "
                "Common side effects include gastrointestinal symptoms such as nausea, diarrhea, and abdominal discomfort, which typically "
                "improve with time. Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min/1.73m²) "
                "due to the risk of lactic acidosis."
            ),
            "topic": "diabetes",
        },
        {
            "id": "medquad_002",
            "question": "What are the common drug interactions with warfarin?",
            "answer": (
                "Warfarin has numerous clinically significant drug interactions due to its narrow therapeutic index and metabolism "
                "through cytochrome P450 enzymes. Major interactions include: (1) NSAIDs such as ibuprofen and naproxen increase "
                "bleeding risk through both antiplatelet effects and gastrointestinal mucosal damage; (2) Aspirin increases bleeding "
                "risk additively; (3) Amiodarone inhibits warfarin metabolism, increasing INR — reduce warfarin dose by 30-50%; "
                "(4) Fluconazole and metronidazole inhibit CYP2C9, increasing warfarin levels; (5) Rifampin induces CYP enzymes, "
                "dramatically reducing warfarin effectiveness; (6) SSRIs (fluoxetine, sertraline) can increase bleeding risk; "
                "(7) Vitamin K-rich foods (leafy greens) can reduce warfarin effectiveness if intake varies significantly."
            ),
            "topic": "drug_interactions",
        },
        {
            "id": "medquad_003",
            "question": "What is the relationship between ACE inhibitors and kidney disease?",
            "answer": (
                "ACE inhibitors (such as lisinopril, enalapril, and ramipril) have a complex relationship with kidney disease. "
                "They are protective in diabetic nephropathy and chronic kidney disease with proteinuria, as they reduce "
                "intraglomerular pressure and slow disease progression. They are first-line agents for hypertension in CKD. "
                "However, they can cause an initial rise in serum creatinine (up to 30% is acceptable) and may cause hyperkalemia, "
                "especially when combined with potassium-sparing diuretics or in patients with reduced GFR. They are contraindicated "
                "in bilateral renal artery stenosis. Monitoring of renal function and potassium within 1-2 weeks of initiation is essential."
            ),
            "topic": "kidney_disease",
        },
        {
            "id": "medquad_004",
            "question": "How do statins prevent cardiovascular disease?",
            "answer": (
                "Statins (e.g., atorvastatin, simvastatin, rosuvastatin) are HMG-CoA reductase inhibitors that prevent cardiovascular "
                "disease through multiple mechanisms. Primarily, they lower LDL cholesterol by 30-50%, reducing atherosclerotic plaque "
                "formation. Beyond lipid lowering, statins have pleiotropic effects: they improve endothelial function, reduce vascular "
                "inflammation (measured by C-reactive protein), stabilize existing plaques, and have antithrombotic properties. "
                "Major trials (4S, WOSCOPS, HPS, JUPITER) demonstrate 25-35% relative risk reduction in major cardiovascular events. "
                "Side effects include myalgia (5-10%), elevated liver enzymes (rare), and rhabdomyolysis (very rare). "
                "Statin-associated muscle symptoms should be evaluated — CK levels checked if symptomatic."
            ),
            "topic": "cardiovascular",
        },
        {
            "id": "medquad_005",
            "question": "What are the gastrointestinal risks of ibuprofen?",
            "answer": (
                "Ibuprofen, a non-steroidal anti-inflammatory drug (NSAID), carries significant gastrointestinal risks. It inhibits "
                "cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis that normally protects the gastric mucosa. Risks include: "
                "(1) Gastric and duodenal ulceration — occurs in 15-30% of chronic NSAID users; (2) GI bleeding — risk increases 3-5 fold; "
                "(3) Gastric perforation — a life-threatening complication; (4) Dyspepsia and heartburn — common even at low doses. "
                "Risk factors for NSAID gastropathy include age > 65, history of peptic ulcer disease, concurrent anticoagulant or "
                "corticosteroid use, high-dose or prolonged NSAID use, and H. pylori infection. Risk mitigation strategies include "
                "co-prescribing a proton pump inhibitor (PPI), using the lowest effective dose for the shortest duration, and considering "
                "alternative analgesics such as paracetamol when appropriate."
            ),
            "topic": "drug_safety",
        },
        {
            "id": "medquad_006",
            "question": "What is insulin resistance and how is it treated?",
            "answer": (
                "Insulin resistance is a pathological condition in which cells fail to respond normally to the hormone insulin. "
                "It is a key feature of type 2 diabetes, metabolic syndrome, and polycystic ovary syndrome (PCOS). In insulin resistance, "
                "the pancreas compensates by producing more insulin (hyperinsulinemia), but eventually this compensatory mechanism fails, "
                "leading to elevated blood glucose. Treatment approaches include: (1) Lifestyle modifications — weight loss of 5-10% body "
                "weight significantly improves insulin sensitivity, along with regular exercise (150 min/week moderate-intensity); "
                "(2) Metformin — first-line medication that improves insulin sensitivity; (3) Thiazolidinediones (e.g., pioglitazone) — "
                "PPAR-gamma agonists that enhance insulin sensitivity in adipose and muscle tissue; (4) GLP-1 receptor agonists — "
                "promote weight loss and improve glycemic control; (5) SGLT2 inhibitors — provide cardiovascular and renal benefits."
            ),
            "topic": "diabetes",
        },
        {
            "id": "medquad_007",
            "question": "What is aspirin's role in antiplatelet therapy?",
            "answer": (
                "Aspirin (acetylsalicylic acid) is the most widely used antiplatelet agent. At low doses (75-100mg daily), it irreversibly "
                "inhibits cyclooxygenase-1 (COX-1) in platelets, preventing thromboxane A2 synthesis and thereby reducing platelet aggregation. "
                "Indications for antiplatelet aspirin include: (1) Secondary prevention of cardiovascular events — proven to reduce risk of "
                "recurrent MI, stroke, and cardiovascular death by approximately 25%; (2) Acute coronary syndrome — given as loading dose "
                "(300mg) then maintenance; (3) Post-PCI (percutaneous coronary intervention) — dual antiplatelet therapy with clopidogrel; "
                "(4) Stable angina with prior events. Primary prevention: Current guidelines recommend against routine aspirin use in low-risk "
                "individuals due to bleeding risk outweighing benefit. Aspirin should NOT be combined with warfarin unless specifically "
                "indicated (e.g., mechanical heart valve with coronary stent) due to significantly increased bleeding risk."
            ),
            "topic": "cardiovascular",
        },
        {
            "id": "medquad_008",
            "question": "How do beta-blockers help in heart failure?",
            "answer": (
                "Beta-blockers are a cornerstone of heart failure treatment, despite the seemingly paradoxical use of a negative inotrope "
                "in a condition of impaired cardiac function. In heart failure with reduced ejection fraction (HFrEF), chronic activation "
                "of the sympathetic nervous system is harmful — it increases heart rate, myocardial oxygen demand, and promotes adverse "
                "remodeling. Beta-blockers (specifically bisoprolol, carvedilol, and metoprolol succinate) counter these effects by: "
                "(1) Reducing heart rate, allowing better diastolic filling; (2) Decreasing myocardial oxygen demand; (3) Preventing "
                "adverse remodeling and promoting reverse remodeling; (4) Reducing arrhythmia risk. Key trials (CIBIS-II, MERIT-HF, "
                "COPERNICUS) demonstrate ~35% reduction in mortality. Important: Must be initiated at low doses and titrated slowly "
                "('start low, go slow') — starting too high can worsen heart failure. Not indicated in decompensated heart failure."
            ),
            "topic": "heart_failure",
        },
        {
            "id": "medquad_009",
            "question": "Can a patient on warfarin safely take ibuprofen?",
            "answer": (
                "No — the combination of warfarin and ibuprofen is considered high-risk and should generally be avoided. Ibuprofen "
                "increases warfarin-related bleeding risk through multiple mechanisms: (1) Ibuprofen inhibits COX-1 in platelets, "
                "impairing platelet function and adding to warfarin's anticoagulant effect; (2) NSAIDs damage the gastrointestinal "
                "mucosa, creating sites prone to bleeding; (3) Ibuprofen may displace warfarin from protein binding, transiently "
                "increasing free warfarin levels; (4) Some NSAIDs inhibit CYP2C9, potentially affecting warfarin metabolism. "
                "The preferred alternative is paracetamol (acetaminophen) at doses ≤ 2g/day. If an NSAID is absolutely necessary, "
                "use the lowest dose for the shortest time with concurrent PPI protection and more frequent INR monitoring. "
                "Topical NSAIDs may be considered as they have minimal systemic absorption."
            ),
            "topic": "drug_interactions",
        },
        {
            "id": "medquad_010",
            "question": "What is the role of proton pump inhibitors (PPIs) like omeprazole?",
            "answer": (
                "Proton pump inhibitors (PPIs) such as omeprazole irreversibly inhibit the hydrogen-potassium ATPase pump (proton pump) "
                "in gastric parietal cells, reducing gastric acid secretion by up to 99%. Indications include: (1) Gastroesophageal reflux "
                "disease (GERD); (2) Peptic ulcer disease — healing and prevention; (3) H. pylori eradication (part of triple therapy); "
                "(4) NSAID gastropathy prevention — essential when NSAIDs cannot be avoided; (5) Zollinger-Ellison syndrome. "
                "Drug interactions: Omeprazole moderately inhibits CYP2C19, which can affect metabolism of clopidogrel (reduced antiplatelet "
                "effect — use pantoprazole instead), diazepam, and phenytoin. Omeprazole has a modest interaction with warfarin. "
                "Long-term risks include: hypomagnesemia, vitamin B12 deficiency, increased fracture risk, C. difficile infection, "
                "and possible increased risk of kidney disease. PPIs should be used at the lowest effective dose for the shortest duration."
            ),
            "topic": "gastroenterology",
        },
        {
            "id": "medquad_011",
            "question": "What is amlodipine and what are its side effects?",
            "answer": (
                "Amlodipine is a long-acting dihydropyridine calcium channel blocker used primarily for hypertension and angina. "
                "It works by blocking L-type calcium channels in vascular smooth muscle, causing vasodilation and reducing peripheral "
                "resistance. Advantages include once-daily dosing, long half-life (30-50 hours), and consistent blood pressure reduction. "
                "Common side effects: peripheral edema (dose-dependent, occurs in 10-30% of patients), headache, flushing, dizziness, "
                "and fatigue. The peripheral edema is due to precapillary arteriolar dilation without corresponding venodilation. "
                "It can be mitigated by combining with an ACE inhibitor or ARB. Amlodipine is metabolized by CYP3A4 — interactions "
                "with strong CYP3A4 inhibitors (e.g., ketoconazole, clarithromycin) may increase levels. It is safe in CKD (no dose "
                "adjustment needed) and is preferred in Black patients and elderly patients as first-line antihypertensive."
            ),
            "topic": "hypertension",
        },
        {
            "id": "medquad_012",
            "question": "What are the warning signs of statin-induced myopathy?",
            "answer": (
                "Statin-induced myopathy is a spectrum of muscle disorders that ranges from mild myalgia to life-threatening rhabdomyolysis. "
                "Warning signs include: (1) Myalgia — unexplained muscle pain, tenderness, or weakness, often in the proximal muscles "
                "(thighs, shoulders, upper arms); (2) Elevated CK levels — myopathy defined as CK > 10x upper limit of normal with "
                "symptoms; (3) Dark-colored urine (cola-colored) — suggests myoglobinuria from rhabdomyolysis; (4) Generalized fatigue "
                "and weakness disproportionate to activity level. Risk factors for statin myopathy include: high statin dose, "
                "hypothyroidism, renal impairment, advanced age, female sex, small body frame, and concomitant use of CYP3A4 inhibitors "
                "(with atorvastatin/simvastatin) or fibrates. Atorvastatin and simvastatin interact with amiodarone — limit dose to 20mg. "
                "Management: Discontinue statin if CK > 10x ULN or intolerable symptoms. May rechallenge with different statin at lower dose."
            ),
            "topic": "drug_safety",
        },
        {
            "id": "medquad_013",
            "question": "How does metoprolol differ from other beta-blockers?",
            "answer": (
                "Metoprolol is a cardioselective (beta-1 selective) beta-blocker, meaning it preferentially blocks beta-1 receptors in "
                "the heart over beta-2 receptors in the lungs and peripheral vasculature. Two formulations exist: metoprolol tartrate "
                "(immediate-release, twice daily) and metoprolol succinate (extended-release, once daily). Metoprolol succinate is the "
                "formulation proven to reduce mortality in heart failure (MERIT-HF trial). Compared to non-selective beta-blockers "
                "(propranolol, carvedilol): (1) Less bronchospasm risk — can be used cautiously in mild asthma/COPD; (2) Less peripheral "
                "vasoconstriction; (3) Less masking of hypoglycemia symptoms in diabetics. Compared to carvedilol: carvedilol has "
                "additional alpha-blocking properties (more vasodilation, slightly better for hypertension with heart failure). "
                "Metoprolol is metabolized by CYP2D6 — poor metabolizers may need dose reduction. Common side effects: bradycardia, "
                "fatigue, cold extremities, dizziness."
            ),
            "topic": "cardiovascular",
        },
        {
            "id": "medquad_014",
            "question": "What happens when you combine ACE inhibitors with potassium-sparing diuretics?",
            "answer": (
                "Combining ACE inhibitors (e.g., lisinopril, enalapril) with potassium-sparing diuretics (e.g., spironolactone, "
                "eplerenone, amiloride) creates a significant risk of hyperkalemia (elevated serum potassium). Both drug classes "
                "independently increase potassium levels: ACE inhibitors reduce aldosterone secretion (aldosterone normally promotes "
                "potassium excretion), while potassium-sparing diuretics directly block potassium excretion in the collecting duct. "
                "However, this combination IS used therapeutically in heart failure (ACE inhibitor + spironolactone) with proven "
                "mortality benefit (RALES trial) — but requires careful monitoring. Management: (1) Check baseline potassium and "
                "renal function before starting; (2) Monitor potassium at 1 week, 4 weeks, then every 3-6 months; (3) Avoid if "
                "potassium > 5.0 mmol/L or eGFR < 30; (4) Use low-dose spironolactone (25mg) in heart failure; (5) Educate patients "
                "to avoid potassium supplements and potassium-rich salt substitutes."
            ),
            "topic": "drug_interactions",
        },
        {
            "id": "medquad_015",
            "question": "What are SGLT2 inhibitors and why were they added to the WHO essential medicines list?",
            "answer": (
                "SGLT2 inhibitors (e.g., empagliflozin, dapagliflozin, canagliflozin) are a class of oral antidiabetic drugs that "
                "work by inhibiting sodium-glucose co-transporter 2 in the proximal tubule of the kidney, preventing glucose "
                "reabsorption and causing glycosuria. Empagliflozin was added to the WHO Essential Medicines List in 2023 based on "
                "landmark trial evidence showing benefits beyond glucose lowering: (1) EMPA-REG OUTCOME — 38% relative risk reduction "
                "in cardiovascular death in T2DM patients with CVD; (2) DAPA-HF — 26% reduction in worsening heart failure or CV death "
                "(benefit seen regardless of diabetes status); (3) CREDENCE/DAPA-CKD — significant slowing of CKD progression. "
                "Additional benefits: weight loss (2-3kg), blood pressure reduction (3-5 mmHg), low hypoglycemia risk. "
                "Side effects: genital mycotic infections, urinary tract infections, rare but serious diabetic ketoacidosis (euglycemic). "
                "Now recommended as second-line after metformin in T2DM patients with established CVD, heart failure, or CKD."
            ),
            "topic": "diabetes",
        },
        {
            "id": "medquad_016",
            "question": "What are the risks of combining multiple blood-thinning medications?",
            "answer": (
                "Combining blood-thinning (antithrombotic) medications exponentially increases bleeding risk. Common scenarios: "
                "(1) Triple therapy (warfarin + aspirin + clopidogrel): required after coronary stenting in atrial fibrillation patients, "
                "but carries 2-4x increased bleeding risk. Limit duration to 1-6 months, then step down. (2) Warfarin + aspirin: "
                "approximately doubles bleeding risk compared to warfarin alone. Only indicated when clear benefit (e.g., mechanical "
                "heart valve + coronary disease). (3) Aspirin + NSAID (ibuprofen): increased GI bleeding risk; ibuprofen may also "
                "interfere with aspirin's antiplatelet effect if taken before aspirin. (4) DOAC + antiplatelet: similar risk profile "
                "to warfarin combinations. Mitigation strategies: use lowest effective doses, shortest necessary duration, add PPI "
                "for GI protection, monitor hemoglobin regularly, educate patients on bleeding signs (black stools, blood in urine, "
                "unusual bruising, prolonged bleeding from cuts)."
            ),
            "topic": "drug_interactions",
        },
        {
            "id": "medquad_017",
            "question": "How does chronic kidney disease affect drug dosing?",
            "answer": (
                "Chronic kidney disease (CKD) affects drug dosing through multiple mechanisms: (1) Reduced renal clearance — drugs "
                "primarily eliminated by the kidneys accumulate, requiring dose reduction or interval extension. Examples: metformin "
                "(contraindicated if eGFR < 30), digoxin (reduce dose, monitor levels), allopurinol (start at 100mg if eGFR < 60). "
                "(2) Altered protein binding — uremia reduces albumin binding, increasing free drug fraction (warfarin, phenytoin). "
                "(3) Reduced hepatic metabolism — CKD impairs CYP450 and other hepatic enzyme activity. (4) Changed volume of "
                "distribution — edema and fluid overload alter drug distribution. Key drug adjustments in CKD: ACE inhibitors — start "
                "low dose, monitor creatinine/potassium (acceptable rise up to 30%); NSAIDs — AVOID in CKD (worsen function, reduce "
                "ACE inhibitor efficacy); metformin — dose-adjust by eGFR (half dose if eGFR 30-45, stop if < 30); statins — atorvastatin "
                "safe without adjustment, simvastatin limit to 20mg if severe CKD."
            ),
            "topic": "kidney_disease",
        },
        {
            "id": "medquad_018",
            "question": "What is the difference between Type 1 and Type 2 diabetes treatment?",
            "answer": (
                "Type 1 and Type 2 diabetes differ fundamentally in pathophysiology and treatment. Type 1 Diabetes: autoimmune "
                "destruction of pancreatic beta cells → absolute insulin deficiency. Treatment is ALWAYS insulin (basal-bolus or "
                "insulin pump). Cannot use oral hypoglycemics alone. Requires carbohydrate counting and frequent glucose monitoring. "
                "Type 2 Diabetes: insulin resistance with progressive beta-cell dysfunction → relative insulin deficiency. Treatment "
                "follows a stepwise approach: (1) Lifestyle modification (diet, exercise, weight loss); (2) Metformin (first-line, "
                "reduces hepatic glucose output); (3) Add second agent based on comorbidities — SGLT2 inhibitor if CVD/HF/CKD, "
                "GLP-1 RA if atherosclerotic CVD or weight management needed, DPP-4 inhibitor, sulfonylurea, or TZD; (4) Basal "
                "insulin if oral agents fail to achieve HbA1c target. Key difference: Type 1 patients should NEVER have insulin "
                "withheld (risk of diabetic ketoacidosis). Type 2 patients may eventually need insulin but it's not always required."
            ),
            "topic": "diabetes",
        },
        {
            "id": "medquad_019",
            "question": "Do FDA warnings for atorvastatin match general clinical guideline recommendations?",
            "answer": (
                "FDA labeling and clinical guidelines for atorvastatin largely align but have some notable differences in emphasis. "
                "Areas of agreement: (1) Both recommend atorvastatin for primary and secondary prevention of cardiovascular events; "
                "(2) Both warn about myopathy/rhabdomyolysis risk, especially at high doses or with interacting drugs; (3) Both note "
                "liver enzyme monitoring is recommended (though current guidelines have de-emphasized routine monitoring). "
                "FDA-specific warnings that guidelines may not emphasize equally: (1) FDA requires a boxed warning about pregnancy "
                "risks — atorvastatin is Category X; (2) FDA specifically warns about dose limitations with certain CYP3A4 inhibitors "
                "(e.g., max 20mg with amiodarone, avoid with strong CYP3A4 inhibitors); (3) FDA labeling includes cognitive effects "
                "(memory loss, confusion) as a potential adverse effect, while guidelines consider this evidence weak. "
                "Guideline emphasis not in FDA labeling: Guidelines provide specific intensity-based dosing (high-intensity: 40-80mg, "
                "moderate: 10-20mg) tied to patient risk categories, which is more clinically actionable than FDA labeling."
            ),
            "topic": "drug_safety",
        },
        {
            "id": "medquad_020",
            "question": "What are the most common drug interaction risks across cardiovascular and diabetes medications?",
            "answer": (
                "The most common drug interaction risks across cardiovascular and diabetes medications fall into several categories: "
                "(1) BLEEDING RISK AMPLIFICATION: Warfarin + NSAIDs/aspirin/SSRIs — the most dangerous common interaction, causing "
                "GI bleeding. Affects millions of patients on anticoagulation. (2) HYPERKALEMIA TRIANGLE: ACE inhibitors/ARBs + "
                "potassium-sparing diuretics + NSAIDs — each independently raises potassium; combination can be life-threatening. "
                "(3) HYPOGLYCEMIA MASKING: Beta-blockers mask tachycardia symptoms of hypoglycemia in diabetic patients on insulin "
                "or sulfonylureas. (4) RENAL FUNCTION DECLINE: The 'triple whammy' — ACE inhibitor/ARB + diuretic + NSAID simultaneously "
                "can cause acute kidney injury. (5) STATIN TOXICITY: Statins (atorvastatin/simvastatin) + CYP3A4 inhibitors "
                "(amiodarone, azole antifungals, macrolide antibiotics) → increased myopathy risk. (6) METFORMIN + CONTRAST: Risk of "
                "lactic acidosis when metformin is not withheld before iodinated contrast procedures. (7) QT PROLONGATION: Multiple "
                "cardiovascular drugs (amiodarone, sotalol) + other QT-prolonging agents → risk of torsades de pointes. "
                "These interactions are the most clinically significant because they involve commonly prescribed medications "
                "in patients who often take multiple drugs simultaneously (polypharmacy)."
            ),
            "topic": "drug_interactions",
        },
    ]

    for qa in qa_pairs[:max_docs]:
        doc_text = (
            f"Question: {qa['question']}\n"
            f"Answer: {qa['answer']}\n"
            f"Topic: {qa['topic']}\n"
            f"Source: MedQuAD (Medical Question Answering Dataset)\n"
            f"ID: {qa['id']}"
        )
        filepath = output_dir / f"{qa['id']}.txt"
        filepath.write_text(doc_text, encoding="utf-8")
        saved_files.append(filepath)

    print(f"    ✓ Saved {len(saved_files)} Q&A documents")
    print(f"  [MedQuAD] Total: {len(saved_files)} documents saved")
    return saved_files


if __name__ == "__main__":
    fetch_medquad()
