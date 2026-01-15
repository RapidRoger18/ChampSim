# Trace List Usage

This folder contains predefined `.tlist` files that are used to run
different categories of trace-driven simulations.
Each file is intended for a specific **evaluation stage** and **system configuration**.

---

## Single-Core Evaluation

Use these files with a **single-core configuration**.

### Quick Debug / Sanity Checks
- `SingleCore_tiny.tlist`  
  Use for fast runs to verify correctness and basic functionality.

### Design Exploration and Tuning
- `SingleCore_small.tlist`  
  Primary set for rapid iteration and parameter tuning.
- `SingleCore_medium.tlist`  
  Use when more stable results are needed without full runtime cost.

### Final Single-Core Results
- `SingleCore_full.tlist`  
  Use for reporting final single-core performance results.

---

## Application Workloads

Use these files with a **single-core configuration** unless stated otherwise.

### PARSEC
- `parsec_small.tlist`  
  Faster PARSEC runs for exploration and debugging.
- `parsec_full.tlist`  
  Final PARSEC evaluation and comparison against prior work.

### Ligra
- `ligra_small.tlist`  
  Smaller graph workloads for early analysis.
- `ligra_full.tlist`  
  Full-scale graph workloads for stress-testing irregular access behavior.

---

## Multicore Evaluation

Use these files **only after finalizing the design in single-core**.

- `MultiCore4_full.tlist`  
  Final evaluation on a **4-core configuration** to assess robustness under
  cache and memory contention.

> Do not use this file for tuning or exploratory analysis.

---

## Recommended Workflow

1. Validate and tune using `SingleCore_small` and `SingleCore_medium`
2. Report single-core results using `SingleCore_full`
3. Validate multicore behavior using `MultiCore4_full`

---

## Notes

- Ensure the simulator configuration matches the intended core count.
- All trace paths listed in these files must exist and be accessible.