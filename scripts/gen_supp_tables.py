#!/usr/bin/env python3
"""Generate Supplementary Tables S8 and S9 for translational vs discovery comparison."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load data
trans_data = {}
disc_data = {}
trans_targets = Counter()
disc_targets = Counter()

with open(ROOT / "results/triple_combinations.csv") as f:
    for row in csv.DictReader(f):
        ct = row["Cancer_Type"]
        targets = [row["Target_1"].strip(), row["Target_2"].strip(), row["Target_3"].strip()]
        drugs = [row.get("Drug_1","N/A"), row.get("Drug_2","N/A"), row.get("Drug_3","N/A")]
        trans_data[ct] = {"targets": targets, "drugs": drugs, "druggable": row.get("Druggable_Count",""), "score": row.get("Combined_Score","")}
        for t in targets:
            trans_targets[t] += 1

with open(ROOT / "results_discovery/triple_combinations.csv") as f:
    for row in csv.DictReader(f):
        ct = row["Cancer_Type"]
        targets = [row["Target_1"].strip(), row["Target_2"].strip(), row["Target_3"].strip()]
        drugs = [row.get("Drug_1","N/A"), row.get("Drug_2","N/A"), row.get("Drug_3","N/A")]
        disc_data[ct] = {"targets": targets, "drugs": drugs, "druggable": row.get("Druggable_Count",""), "score": row.get("Combined_Score","")}
        for t in targets:
            disc_targets[t] += 1

# ── Table S8: Per-cancer translational vs discovery comparison ──
shared_cancers = sorted(set(trans_data) & set(disc_data))
with open(ROOT / "supplementary_tables/Table_S8_translational_vs_discovery.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Cancer_Type", "Translational_Triple", "Translational_Drugs", "Trans_Druggable",
                "Discovery_Triple", "Discovery_Drugs", "Disc_Druggable",
                "Shared_Targets", "Agreement"])
    for ct in shared_cancers:
        t = trans_data[ct]
        d = disc_data[ct]
        t_set = set(t["targets"])
        d_set = set(d["targets"])
        overlap = t_set & d_set
        n = len(overlap)
        if n == 3:
            status = "IDENTICAL"
        elif n == 0:
            status = "DIFFERENT"
        else:
            status = f"{n}/3 overlap"
        w.writerow([
            ct,
            " + ".join(sorted(t["targets"])),
            " + ".join(t["drugs"]),
            t["druggable"],
            " + ".join(sorted(d["targets"])),
            " + ".join(d["drugs"]),
            d["druggable"],
            " + ".join(sorted(overlap)) if overlap else "none",
            status,
        ])

# ── Table S9: Drug development priorities (discovery-only targets) ──
trans_set = set(trans_targets)
disc_set = set(disc_targets)
disc_only = disc_set - trans_set

with open(ROOT / "supplementary_tables/Table_S9_discovery_drug_priorities.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Target", "Discovery_Frequency", "Cancer_Types", "Druggability_Status", "Modality_Opportunity"])
    for gene in sorted(disc_only, key=lambda x: -disc_targets[x]):
        freq = disc_targets[gene]
        # Find which cancers
        cancers = []
        for ct, d in disc_data.items():
            if gene in d["targets"]:
                cancers.append(ct)
        # Classify
        known_druggable = {"CCND1": "Indirect (CDK4/6i)", "CTNNB1": "Undruggable (WNT pathway)",
                           "YAP1": "Undruggable (Hippo pathway)", "GRB2": "PPI inhibitor (preclinical)",
                           "PTK2": "Phase 2 (defactinib)", "E2F3": "Undruggable (TF)",
                           "PXN": "Undruggable (scaffold)", "FOXM1": "Phase 1 (FDI-6)",
                           "CCNE1": "Indirect (CDK2i)", "CCNE2": "Indirect (CDK2i)",
                           "MAPK1": "Approved (ulixertinib P2)", "MAPK3": "Approved (ulixertinib P2)",
                           "STAT5A": "Phase 2 (pimozide)", "GAB2": "Undruggable (adaptor)",
                           "SOX10": "Undruggable (TF)", "LCK": "Approved (dasatinib)",
                           "SLC2A1": "Phase 1 (BAY-876)", "FOXA1": "Undruggable (TF)"}
        modality_map = {"CCND1": "PROTAC/CDK4/6i", "YAP1": "PROTAC/peptide",
                        "GRB2": "PROTAC/SH2 inhibitor", "PTK2": "Small molecule (P2)",
                        "E2F3": "PROTAC", "CTNNB1": "PROTAC/stapled peptide",
                        "PXN": "PROTAC", "CCNE1": "CDK2i (seliciclib)", "CCNE2": "CDK2i",
                        "STAT5A": "PROTAC/JAK inhibitor"}
        status = known_druggable.get(gene, "No clinical inhibitor")
        modality = modality_map.get(gene, "Novel target — needs drug discovery")
        w.writerow([gene, freq, "; ".join(sorted(cancers)[:5]) + ("..." if len(cancers)>5 else ""), status, modality])

print("Generated:")
print("  supplementary_tables/Table_S8_translational_vs_discovery.csv")
print("  supplementary_tables/Table_S9_discovery_drug_priorities.csv")
