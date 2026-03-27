# XGBoost: Why Gradient Boosting Wins

A complete, step-by-step Jupyter notebook tutorial on XGBoost using NBA 2023–24 player statistics — covering the mathematics of gradient boosting, head-to-head model comparison, hyperparameter deep-dives, and feature importance analysis.

---

## What This Notebook Covers

This tutorial answers one central question: **why does XGBoost consistently outperform simpler models on tabular data?**

It answers through three lenses:

1. **Mathematical** — how gradient boosting sequentially corrects errors tree by tree
2. **Comparative** — Decision Tree vs Random Forest vs XGBoost side by side
3. **Hyperparameter deep-dive** — how `n_estimators`, `learning_rate`, and `max_depth` interact and what each actually does

**Task:** Predict NBA player position (Guard / Forward / Centre) from per-game statistics.

| Step | Topic |
|------|-------|
| 0 | Install packages |
| 1 | Imports and settings |
| 2 | Load and explore the NBA dataset |
| 3 | Prepare features and split data |
| 4 | The mathematics of gradient boosting |
| 5 | Figure 1: Model comparison baseline |
| 6 | Figure 2: `n_estimators` sweep |
| 7 | Figure 3: `learning_rate` sweep |
| 8 | Figure 4: `max_depth` sweep |
| 9 | Figure 5: Hyperparameter interaction heatmap |
| 10 | Figure 6: Feature importance |
| 11 | Final evaluation and classification report |
| 12 | Key takeaways and extensions |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/xgboost-nba-tutorial.git
cd xgboost-nba-tutorial
```

### 2. Install dependencies

Python 3.8+ is required.

```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

> **macOS users:** XGBoost requires OpenMP. Install it via Homebrew before running the notebook:
> ```bash
> brew install libomp
> ```

### 3. Open the notebook

```bash
jupyter notebook xgboost_nba_tutorial.ipynb
```

Run cells from top to bottom. The dataset is constructed inline — no external files or downloads needed.

---

## Dataset

**NBA 2023–24 per-game statistics** sourced from [Basketball Reference](https://www.basketball-reference.com/leagues/NBA_2024_per_game.html). The dataset is embedded directly in the notebook as a Python dictionary, so no download or API key is required.

The 5 NBA positions are simplified into 3 target classes:

| Class | Positions | Description |
|-------|-----------|-------------|
| Guard (G) | PG, SG | Ball-handlers, shooters |
| Forward (F) | SF, PF | Versatile scorers |
| Centre (C) | C | Interior players, rebounders |

Position prediction is a good XGBoost task because positions overlap in modern basketball — no single stat perfectly separates them.

### Features used

| Feature | Description |
|---------|-------------|
| `G` | Games played |
| `MP` | Minutes per game |
| `PTS` | Points per game |
| `TRB` | Total rebounds per game |
| `AST` | Assists per game |
| `STL` | Steals per game |
| `BLK` | Blocks per game |
| `TOV` | Turnovers per game |
| `FG_PCT` | Field goal percentage |
| `P3_PCT` | Three-point percentage |
| `FT_PCT` | Free throw percentage |
| `ORB` | Offensive rebounds per game |
| `PF` | Personal fouls per game |

**Train/test split:** 80% train / 20% test, stratified by class. Model selection uses 5-fold cross-validation on the training set.

---

## The Mathematics

XGBoost builds an ensemble sequentially, where each new tree corrects the residuals of all previous trees:

```
F_0(x) = initial prediction
F_1(x) = F_0(x) + lr * tree_1(x)    ← tree_1 fits residuals of F_0
F_2(x) = F_1(x) + lr * tree_2(x)    ← tree_2 fits residuals of F_1
...
F_M(x) = F_{M-1}(x) + lr * tree_M(x)
```

XGBoost extends standard gradient boosting with four key improvements (Chen & Guestrin, 2016):

| Improvement | What it does | Why it matters |
|-------------|-------------|----------------|
| Regularisation (λ, α) | Penalises tree complexity | Prevents overfitting |
| Second-order gradients | Uses curvature of the loss function | Faster, more accurate updates |
| Column subsampling | Random feature subsets per tree | Reduces inter-tree correlation |
| Parallelised tree building | Sorts all features simultaneously | 10–100× faster than sklearn GBM |

---

## Models Compared

| Model | Configuration |
|-------|--------------|
| Decision Tree | Default sklearn, `random_state=42` |
| Random Forest | 100 estimators, `random_state=42` |
| XGBoost | 100 estimators, default hyperparameters, `eval_metric=mlogloss` |

All models evaluated with 5-fold cross-validation on the training set, then assessed on the held-out test set.

---

## Hyperparameters Explored

| Hyperparameter | Values swept | What it controls |
|----------------|-------------|-----------------|
| `n_estimators` | 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500 | Number of sequential trees |
| `learning_rate` | 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 | Step size shrinkage per tree |
| `max_depth` | 1, 2, 3, 4, 5, 6, 7, 8, 10, 12 | Maximum depth of each tree |

A full grid search over `learning_rate × max_depth` is also run and visualised as a heatmap (Figure 5) to show how hyperparameters interact.

### Key rules of thumb

| Hyperparameter | Typical best range | Effect of too high | Effect of too low |
|----------------|-------------------|-------------------|-------------------|
| `n_estimators` | 100–500 | Overfitting + slow | Underfitting |
| `learning_rate` | 0.05–0.2 | Overfitting | Needs many more trees |
| `max_depth` | 3–6 | Overfitting | Too simple, underfitting |

**The golden rule:** lower `learning_rate` + higher `n_estimators` almost always improves generalisation, at the cost of training time.

---

## Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Model comparison — test accuracy of Decision Tree, Random Forest, and XGBoost |
| Fig 2 | `n_estimators` sweep — train vs test accuracy across 12 values |
| Fig 3 | `learning_rate` sweep — train vs test accuracy across 10 values |
| Fig 4 | `max_depth` sweep — train vs test accuracy across 10 depths |
| Fig 5 | Hyperparameter interaction heatmap — `learning_rate × max_depth` grid search |
| Fig 6 | Feature importance — XGBoost's built-in gain scores for all 13 features |

---

## Extensions to Try

- **`subsample`** (0.5–0.8): randomly sample rows per tree — reduces overfitting analogously to Random Forest
- **`colsample_bytree`** (0.5–0.8): randomly sample features per tree — XGBoost's equivalent of Random Forest feature sampling
- **`reg_alpha` / `reg_lambda`**: L1/L2 regularisation — penalises leaf weights directly
- **Early stopping**: pass `eval_set` to automatically stop when validation accuracy plateaus — finds the optimal `n_estimators` without a full grid search

---

## Accessibility

All figures use the **Wong (2011) 8-colour colourblind-safe palette**, distinguishable under deuteranopia, protanopia, and tritanopia. Figure DPI is set to 120 throughout for crisp rendering in notebooks. A fixed random seed (`SEED = 42`) is used throughout for full reproducibility.

---

## References

1. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. https://arxiv.org/abs/1603.02754
2. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 29(5), 1189–1232.
3. Basketball Reference. *2023–24 NBA Per Game Statistics.* https://www.basketball-reference.com/leagues/NBA_2024_per_game.html
4. XGBoost Documentation. https://xgboost.readthedocs.io
5. scikit-learn Documentation. https://scikit-learn.org/stable/
6. Wong, B. (2011). *Color blindness.* Nature Methods, 8(6), 441. https://doi.org/10.1038/nmeth.1618

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
