# Reconciliation Audit: Baseline vs MetaGate Studies

**Date:** April 2026  
**Purpose:** Verify compatibility between corrected baseline report and MetaGate report before merging  
**Auditor:** Automated reconciliation process

---

## Executive Summary

This audit reconciles two experimental bundles:
1. **Baseline Evaluation Study** (`results/final_full_eval_corrected/`)
2. **MetaGate Evaluation Study** (`results/dynamic_metagate/`)

**Finding:** The studies are related but not directly comparable. The merged report must keep results in separate sections with explicit provenance.

---

## Detailed Audit Table

| Item | Baseline Corrected Report | MetaGate Report | Compatible? | Resolution |
|------|---------------------------|-----------------|-------------|------------|
| **Method Scope** | 6 standalone methods | MetaGate + 4 experts | Partially | Keep separate; MetaGate is meta-selector using baseline experts |
| **Number Evaluated** | 6 succeeded, 5 failed | 1 gate + 4 experts complete | Yes | Different scopes; baseline is broader, MetaGate is deeper |
| **Topology Naming** | abilene_backbone, cernet_real, etc. | abilene, cernet, etc. | **No** | Same topologies, different naming conventions; explicit mapping required |
| **CERNET Nodes** | **ERROR: Report stated 20 nodes** | 41 nodes | **Reconciled** | Config (`phase1_reactive_topologies.yaml:87-96`) is authoritative: **41 nodes, 59 bidir links** |
| **CERNET Source** | Not explicitly stated | TopologyZoo, MGM synthetic | **Resolved** | Config confirms TopologyZoo source with synthetic MGM traffic |
| **Germany50 Status** | germany50_real (known) | germany50 (marked "unseen") | Yes | Same topology; MetaGate explicitly marks as generalization test |
| **Germany50 Edges** | 88 links (report) | Not specified | **Consistent** | Config line 30: `expected_num_edges: 176` (directed) = 88 undirected links |
| **VtlWavenet Status** | vtlwavenet2011 | topologyzoo_vtlwavenet2011 | Yes | **CORRECTION:** MetaGate marks VtlWavenet as **UNSEEN** (not "known") |
| **Failure Robustness** | 5 scenarios, MLU valid, disturbance broken | Limited failure analysis | Partially | Use baseline MLU only; disturbance is unreliable |
| **Disturbance** | All 0.0 (impossible) | Not primary metric | **Broken** | Do not interpret baseline disturbance metrics |
| **Timing** | ms precision | ms precision | Yes | Comparable; MetaGate includes gate overhead |
| **Recommendation** | "Heuristics best" | "MetaGate strong on Germany50" | Synthesizable | Different contexts; both valid for their scopes |

---

## Critical Corrections Applied

### 1. CERNET Topology Documentation Error

**Error Identified:** The baseline report incorrectly stated "20 nodes, 32 links" for CERNET.

**Evidence from Config:**
```yaml
# configs/phase1_reactive_topologies.yaml:87-96
- key: cernet_real
  dataset_key: cernet
  display_name: "CERNET (41 nodes, 59 bidirectional links)"
  source: topologyzoo
  expected_num_nodes: 41
  expected_num_edges: 116  # 59 undirected links x 2 = 116 directed edges
```

**Resolution:** Use 41 nodes, 59 bidir links (116 directed edges) as authoritative. Explicitly document the prior error in the merged report.

### 2. VtlWavenet Status Correction

**Initial Audit Error:** Assumed VtlWavenet was "known" in MetaGate.

**Evidence:**
```
# metagate_summary.csv
topologyzoo_vtlwavenet2011,unseen,0.507...
```

**Correction:** VtlWavenet is marked **UNSEEN** in MetaGate, representing a generalization test on the largest topology (92 nodes).

### 3. Germany50 Edge Count Verification

**Question:** Baseline report says "88 links"; is this compatible with config?

**Evidence from Config:**
```yaml
# configs/phase1_reactive_topologies.yaml:30
expected_num_nodes: 50
expected_num_edges: 176  # directed edges
```

**Resolution:** 88 undirected links = 176 directed edges. The baseline report is **correct** when interpreting "links" as undirected. Config confirms this.

### 4. Failure Robustness Limitation

**Finding:** Baseline failure_results.csv has disturbance = 0.0 for all 1,200 rows.

**Evidence:**
```bash
$ awk -F',' 'NR>1 {print $9}' results/final_full_eval_corrected/failure_results.csv | sort | uniq -c
1200 0.0
```

**Root Cause:** Bug in prev_sel persistence across scenarios—timestep indices reset but prev_sel does not, causing incorrect disturbance calculation.

**Resolution:** Use only MLU degradation data from failure scenarios; explicitly mark disturbance as unreliable.

---

## Merge Strategy

### What CAN Be Combined
- Pipeline description (K=40, K=3, LP optimization)
- General methodology framing
- Timing metric formats (both use milliseconds)
- Topology lists (with explicit name mapping)

### What MUST Remain Separate
- **Results tables:** Baseline (6 methods) and MetaGate (1 gate + 4 experts) in separate sections
- **CERNET documentation:** Explicitly note the baseline report error and correction
- **Failure robustness:** Use baseline MLU only; mark disturbance as broken
- **Calibration analysis:** MetaGate-specific (Germany50 before/after)

### Explicit Provenance Required
Every table and figure must indicate its source:
- "From baseline evaluation study"
- "From MetaGate evaluation study"

---

## Sections Using Baseline Material

1. **Executive Summary** — Partial (baseline context)
2. **Evaluation Scope and Provenance** — Baseline study description
3. **Locked TE Pipeline** — Shared methodology
4. **Section 4: Baseline Evaluation Results** — Primary baseline section
5. **Section 8: Failure Robustness** — Conservative use with warnings

## Sections Using MetaGate Material

1. **Executive Summary** — Partial (MetaGate context)
2. **Evaluation Scope and Provenance** — MetaGate study description
3. **Section 5: MetaGate Final Method** — Architecture and training
4. **Section 6: MetaGate Results** — Germany50 calibration, VtlWavenet analysis
5. **Section 7: Cross-Study Synthesis** — Integration point

---

## Topology Name Mapping

| Baseline Name | MetaGate Name | Notes |
|---------------|---------------|-------|
| abilene_backbone | abilene | Same topology, SNDlib source |
| cernet_real | cernet | **41 nodes** (not 20); TopologyZoo source |
| geant_core | geant | Same topology, SNDlib source |
| germany50_real | germany50 | Same topology; MetaGate marks as "unseen" |
| ebone | rocketfuel_ebone | Rocketfuel source |
| sprintlink | rocketfuel_sprintlink | Rocketfuel source |
| tiscali | rocketfuel_tiscali | Rocketfuel source |
| vtlwavenet2011 | topologyzoo_vtlwavenet2011 | **92 nodes, UNSEEN** in MetaGate |

---

## Recommendations for Final Report

1. **Be explicit about provenance** — Every claim must indicate which study it comes from
2. **Correct CERNET documentation** — Use 41 nodes; note the prior error
3. **Mark VtlWavenet as UNSEEN** — In MetaGate context, this is a generalization test
4. **Conservative failure analysis** — Use MLU only; disturbance is unreliable
5. **Honest synthesis** — Heuristics are best for simple deployments; MetaGate adds value for advanced requirements with calibration
6. **No false unified narrative** — The studies are separate bundles; present them as such

---

## Verification Commands

```bash
# Verify baseline methods
cut -d',' -f2 results/final_full_eval_corrected/final_results.csv | sort | uniq -c
# Output: 40 each of Bottleneck, ECMP, GNN, OSPF, Sensitivity, TopK

# Verify MetaGate topology types
cut -d',' -f2 results/dynamic_metagate/metagate_summary.csv
# Output: known/unseen flags per topology

# Verify CERNET config
grep -A 10 "cernet_real" configs/phase1_reactive_topologies.yaml
# Output: 41 nodes, 59 bidir links

# Verify failure disturbance bug
awk -F',' 'NR>1 {print $9}' results/final_full_eval_corrected/failure_results.csv | sort | uniq -c
# Output: 1200 occurrences of 0.0
```

---

## Conclusion

The baseline and MetaGate studies are **separate but complementary**. The merged report must:
- Present each study's results in its own section
- Explicitly map topology naming differences
- Correct the CERNET documentation error
- Use conservative language for failure robustness
- Synthesize honest recommendations based on actual evidence

**Status:** AUDIT COMPLETE — Ready for merged report generation
