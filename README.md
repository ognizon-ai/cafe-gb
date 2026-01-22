# CAFÉ-GB

## Step 1
```bash
python run_all.py --stage cafe --dataset andmal2020
python run_all.py --stage cafe --dataset bodmas
```

## Phase 2 — k-selection 
```bash
python run_all.py --stage k --dataset embod
python run_all.py --stage k --dataset andmal2020
```

## Phase 3 — Classification
```bash
python run_all.py --stage classify --dataset andmal2020
python run_all.py --stage classify --dataset bodmas
```

## Phase 4 — Analysis & stats
```bash
python analysis/correlation_redundancy.py --dataset bodmas --seed 42 --k 100
python analysis/statistical_tests.py --dataset bodmas --metric all
python analysis/shap_analysis.py --dataset bodmas --model LGBM
```