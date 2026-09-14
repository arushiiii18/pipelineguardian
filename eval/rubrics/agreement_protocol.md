# Inter-Rater / Self-Agreement Protocol

## Overview
Ground-truth annotations must be reliable and reproducible. This document specifies the agreement measurement protocol for ground-truth labeling.

---

## 1. Primary Protocol: Independent Dual-Annotation
1. **Sampling**: A random 20% subset of notebooks in the benchmark corpus is selected.
2. **Independent Labeling**: Annotator 1 and Annotator 2 independently label the subset against [`eval/rubrics/reproducibility_rubric.md`](reproducibility_rubric.md) without inspecting each other's labels.
3. **Metrics Reported**:
   - **Percentage Agreement**: $P_o = \frac{\text{Number of agreeing items}}{\text{Total items}}$
   - **Cohen's Kappa ($\kappa$)**:
     $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
     where $P_e$ is the expected agreement by chance.

---

## 2. Secondary Protocol (Fallback): Blind Self-Relabeling
If a second annotator is unavailable:
1. **Time Interval**: The primary annotator waits at least 48 hours after completing initial labeling.
2. **Masked Re-annotation**: The primary annotator re-labels a masked random 20% subset without viewing previous annotations.
3. **Reporting**: The resulting agreement is explicitly reported as **"Blind Self-Relabeling Agreement"** (not inter-rater agreement) to maintain academic honesty.

---

## 3. Mandatory Guideline
> [!CAUTION]
> Do NOT manufacture, estimate, or invent agreement numbers. If dual annotation or self-relabeling has not been executed, report agreement as **"Not Yet Executed (Pending Corpus Population)"**.
