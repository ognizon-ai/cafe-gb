# CAFÉ-GB — Paper 1 Experiments

## Step 1
```bash
python run_all.py --stage cafe --dataset andmal2020
python run_all.py --stage cafe --dataset bodmas
python run_all.py --stage cafe --dataset embod
```

## Phase 2 — k-selection (lightweight, once)
```bash
python run_all.py --stage k --dataset embod
python run_all.py --stage k --dataset andmal2020
python run_all.py --stage k --dataset bodmas
```

## Phase 3 — Classification (moderate)
```bash
python run_all.py --stage classify --dataset andmal2020
python run_all.py --stage classify --dataset bodmas
python run_all.py --stage classify --dataset embod
```

## Phase 4 — Analysis & stats (cheap)
```bash
python run_all.py --stage correlation
python run_all.py --stage stats
python run_all.py --stage shap --dataset embod
python run_all.py --stage shap --dataset andmal2020
python run_all.py --stage shap --dataset bodmas
```

## Run
```bash
# CAFÉ-GB only (heavy)
python run_all.py --stage cafe --dataset andmal2020

# k-selection (once, reference dataset)
python run_all.py --stage k --dataset embod

# Classification (cheap)
python run_all.py --stage classify --dataset bodmas
# Analysis & stats (cheap)



Perfect — you’re at the **right execution point**.
Below is the **exact, safe, step-by-step command sequence** from **this moment onward**, assuming:

* You have **finished CAFÉ-GB for BODMAS**
* You want **final k = 100**
* You want **all experiments completed cleanly**
* You want **NO re-runs and NO wasted compute**

I’ll give you **only commands**, in the **correct order**, with **one-line context**.

---

## 🔴 STEP 1 — Finish CAFÉ-GB for remaining datasets (if not done)

If **only BODMAS is done**, run CAFÉ-GB for the others.

```bash
python run_all.py --stage cafe --dataset andmal2020
python run_all.py --stage cafe --dataset embod
```

✔ Heavy step
✔ Run one dataset at a time
✔ Can be stopped/resumed

---

## 🟡 STEP 2 — k-selection (run ONCE)

You said **k = 100**.
If you **already ran k-selection earlier**, skip this step.

Otherwise, run **only on reference dataset** (recommended: `embod`):

```bash
python run_all.py --stage k --dataset embod
```

👉 Inspect:

```bash
results/tables/embod/table3_k_selection.xlsx
```

✔ Confirm `k = 100`

---

## 🟢 STEP 3 — Lock k = 100 (IMPORTANT)

Edit `config/k.yaml`:

```yaml
k: 100
```

(No command needed if already set.)

---

## 🟢 STEP 4 — Classification (ALL datasets)

Now run classifiers using **fixed k = 100**.

```bash
python run_all.py --stage classify --dataset bodmas
python run_all.py --stage classify --dataset andmal2020
python run_all.py --stage classify --dataset embod
```

✔ Fast compared to CAFÉ-GB
✔ Uses cached features
✔ Produces Tables 6–8

---

## 🔵 STEP 5 — Verify runtime & memory profiling

Check that profiling was logged:

```bash
cat results/tables/runtime_memory.csv
```

✔ Needed for **Scalability Table (Table 5)**

---

## 🔵 STEP 6 — Verify all outputs exist

### CAFÉ-GB outputs

```bash
ls fs/cafe_gb/bodmas
ls fs/cafe_gb/andmal2020
ls fs/cafe_gb/embod
```

You should see:

```
aggregated_importance_seed*.parquet
```

---

### Classification outputs

```bash
ls results/tables/bodmas
ls results/tables/andmal2020
ls results/tables/embod
```

You should see:

```
performance_seed42.xlsx
performance_seed52.xlsx
...
```

---

## 🔵 STEP 7 — Aggregate results (NO more heavy runs)

At this point:

* All **experiments are DONE**
* Only **analysis & plotting** remains

(No more `run_all.py` calls needed.)

---

## 📌 FINAL EXECUTION SUMMARY (copy–paste safe)

```bash
# CAFÉ-GB
python run_all.py --stage cafe --dataset bodmas
python run_all.py --stage cafe --dataset andmal2020
python run_all.py --stage cafe --dataset embod

# k-selection (once)
python run_all.py --stage k --dataset embod

# Fix k = 100 in config/k.yaml

# Classification
python run_all.py --stage classify --dataset bodmas
python run_all.py --stage classify --dataset andmal2020
python run_all.py --stage classify --dataset embod
```

---

## 🎯 You are DONE with computation

From here on:

* 📊 Tables
* 📈 Plots
* ✍️ Paper writing

No more expensive runs.

---

# Rerun Experiments
---

# 🔁 GENERAL RULE (memorize this)

> **You rerun a stage by deleting *only that stage’s outputs*.
> Never delete upstream artifacts unless the change affects them.**

Pipeline dependency is:

```
CAFÉ-GB  →  k-selection  →  Classification  →  Analysis
```

---

# 🟥 STAGE 1 — CAFÉ-GB (Feature Selection)

## When do you need to rerun CAFÉ-GB?

Rerun **ONLY IF you change**:

* chunk_size / overlap (`config/cafe_gb.yaml`)
* feature importance model (GB → LGBM)
* feature preprocessing
* random seeds
* CAFÉ-GB code (`fs/cafe_gb/*`)

### ❌ Do NOT rerun CAFÉ-GB if you only change:

* k
* classifiers
* metrics
* plotting
* saving models

---

## How to rerun CAFÉ-GB (three levels)

### 🔹 A. Rerun CAFÉ-GB for ONE seed (recommended)

```bash
rm fs/cafe_gb/bodmas/aggregated_importance_seed42.parquet
python run_all.py --stage cafe --dataset bodmas
```

Only seed 42 reruns.

---

### 🔹 B. Rerun CAFÉ-GB for ONE dataset (all seeds)

```bash
rm -rf fs/cafe_gb/bodmas
python run_all.py --stage cafe --dataset bodmas
```

---

### 🔹 C. Rerun CAFÉ-GB for ALL datasets

```bash
rm -rf fs/cafe_gb
python run_all.py --stage cafe
```

⚠️ Very expensive — do this only if unavoidable.

---

# 🟨 STAGE 2 — k-selection

## When do you need to rerun k-selection?

Rerun **ONLY IF you change**:

* CAFÉ-GB results
* k-selection logic (`stats/k_selection.py`)
* k candidate list

### ❌ Do NOT rerun if:

* only classifiers changed
* only reporting changed

---

## How to rerun k-selection

### 🔹 A. Rerun for ONE dataset

```bash
rm results/tables/bodmas/table3_k_selection.xlsx
rm -rf results/figures/bodmas
python run_all.py --stage k --dataset bodmas
```

---

### 🔹 B. Rerun for reference dataset only (recommended)

```bash
rm results/tables/embod/table3_k_selection.xlsx
rm -rf results/figures/embod
python run_all.py --stage k --dataset embod
```

---

# 🟩 STAGE 3 — Classification

## When do you need to rerun classification?

Rerun **IF you change**:

* k (`config/k.yaml`)
* classifiers / hyperparameters
* model-saving logic
* metrics
* profiling code

### ❌ Do NOT rerun if:

* CAFÉ-GB unchanged
* k unchanged
* only plotting changes

---

## How to rerun classification

### 🔹 A. Rerun ONE seed (fast, profiling / sanity)

```bash
rm results/tables/bodmas/performance_seed42.xlsx
python run_all.py --stage classify --dataset bodmas
```

---

### 🔹 B. Rerun ONE dataset (all seeds)

```bash
rm -rf results/tables/bodmas
python run_all.py --stage classify --dataset bodmas
```

---

### 🔹 C. Rerun ALL datasets

```bash
rm -rf results/tables
python run_all.py --stage classify
```

---

# 🟦 STAGE 4 — Profiling (runtime & memory)

## When do you need to rerun profiling?

Profiling happens **only when a stage actually runs**.

So:

* If a stage is skipped → no profiling
* To profile → force that stage to rerun

### Recommended profiling strategy

* CAFÉ-GB → 1 seed per dataset
* Classification → 1 seed per dataset

---

### Example: Profile CAFÉ-GB

```bash
rm fs/cafe_gb/bodmas/aggregated_importance_seed42.parquet
python run_all.py --stage cafe --dataset bodmas
```

---

### Example: Profile classification

```bash
rm results/tables/bodmas/performance_seed42.xlsx
python run_all.py --stage classify --dataset bodmas
```

---

# 🟪 STAGE 5 — Analysis / Tables / Plots

## When do you need to rerun analysis?

Rerun **ONLY IF you change**:

* aggregation scripts
* plotting scripts
* statistical tests

### ❌ Do NOT rerun any experiments for this.

Just rerun scripts, e.g.:

```bash
python stats/aggregate_results.py
python plots/make_figures.py
```

(No `run_all.py` involved.)

---

# 📌 QUICK DECISION TABLE

| You changed… | Rerun                        |
| ------------ | ---------------------------- |
| k value      | Classification               |
| CAFÉ-GB code | CAFÉ-GB → k → Classification |
| Classifier   | Classification               |
| Metrics      | Classification               |
| Plots only   | Nothing                      |
| Profiling    | Rerun that stage             |
| Seeds        | CAFÉ-GB + Classification     |

---

# ✅ SAFE DEFAULT (when unsure)

If you’re unsure what changed, this is safe and not too expensive:

```bash
rm -rf results/tables/bodmas
python run_all.py --stage classify --dataset bodmas
```

---

## 🎯 Final advice (important)

* **Delete outputs, never inputs**
* **Rerun the minimum stage**
* **Never rerun CAFÉ-GB unless absolutely required**
* Your current setup is **textbook-correct**





