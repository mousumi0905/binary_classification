# 🚀 Spaceship Titanic — Predicting Dimensional Transport

A binary classification project built on Kaggle's *Spaceship Titanic* dataset, predicting whether a passenger was transported to an alternate dimension during the ship's collision with a spacetime anomaly. The project covers the full applied ML workflow: exploratory data analysis, feature engineering, a `ColumnTransformer`-based preprocessing pipeline, and a comparison of three classification algorithms.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Pandas](https://img.shields.io/badge/Pandas-DataAnalysis-150458)
![Status](https://img.shields.io/badge/Status-Baseline%20Complete-yellow)

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Modeling & Evaluation](#modeling--evaluation)
- [Results](#results)
- [Critical Analysis & Next Steps](#critical-analysis--next-steps)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## 🪐 Problem Statement

In the year 2912, the *Spaceship Titanic* collided with a spacetime anomaly while en route to three newly habitable exoplanets. Roughly half of the ~13,000 passengers were transported to an alternate dimension. Given passenger records — demographics, cabin information, and onboard spending — the task is to predict the binary target **`Transported`** (`True`/`False`) for each passenger in the test set.

This is framed as a **supervised binary classification** problem and evaluated on classification accuracy.

---

## 📊 Dataset

The dataset is provided by Kaggle as part of the *Spaceship Titanic* "Getting Started" competition.

| Split | Rows | Columns |
|-------|------|---------|
| Train | 8,693 | 14 |
| Test  | 4,277 | 13 (no target) |

**Original feature set:**

| Feature | Description |
|---------|--------------|
| `PassengerId` | Unique ID in the form `gggg_pp` — `gggg` is the travel group, `pp` is the passenger number within the group |
| `HomePlanet` | Planet of departure (categorical) |
| `CryoSleep` | Whether the passenger elected suspended animation for the voyage |
| `Cabin` | Cabin in the form `deck/num/side` |
| `Destination` | Planet the passenger is debarking to |
| `Age` | Passenger age |
| `VIP` | Whether the passenger paid for VIP service |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | Billing amounts at each onboard amenity |
| `Name` | Passenger name |
| `Transported` | **Target** — whether the passenger was transported to another dimension |

---

## 🔍 Exploratory Data Analysis

Initial analysis focused on understanding data quality and the distribution of feature types before any modeling decisions were made:

- **Shape & types:** The training set contains 8,693 rows across object, float64, and bool dtypes.
- **Missing data:** Every feature except `PassengerId` and `Transported` has missing values, but each only at the **2–3% level**. Given this is well below the typical 80–95% drop threshold, all columns were retained and handled via imputation rather than removal.
- **Cardinality check:** `describe(include=['O'])` revealed that `PassengerId` and `Name` are unique identifiers for every passenger — they carry no generalizable signal for a classifier and were dropped.
- **Spending behavior:** Summary statistics on `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck` show highly right-skewed distributions (means far above medians, e.g., `FoodCourt` mean ≈ 458 vs. median = 0), indicating most passengers spent little to nothing while a small number of high spenders pull the distribution.

---

## 🛠️ Feature Engineering

Based on the EDA, the following transformations were applied to both train and test sets:

1. **Dropped non-predictive identifiers:** `PassengerId` and `Name` (unique per row, no signal).
2. **Decomposed `Cabin`** into `Deck`, `Num`, and `Side` by splitting on `/`. The `Num` field (cabin number) was dropped due to very high cardinality (1,817 unique values) relative to its likely predictive value, while `Deck` and `Side` were retained as low-cardinality categoricals.
3. **Engineered `total_spent`** as the sum of `RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`, consolidating five sparse, highly-correlated spending columns into a single aggregate feature. The individual amenity columns were then dropped.

**Final modeling feature set (8 features):**
`HomePlanet`, `CryoSleep`, `Destination`, `Age`, `VIP`, `Deck`, `Side`, `total_spent`

---

## ⚙️ Preprocessing Pipeline

A `ColumnTransformer` was used to apply distinct preprocessing strategies to numerical and categorical features, wrapped inside a `scikit-learn` `Pipeline` for reproducibility and to prevent data leakage between train/test splits:

```python
numerical_pipe = Pipeline(steps=[
    ('norm', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('norm', StandardScaler(with_mean=False))
])

preprocessors = ColumnTransformer(transformers=[
    ('num', numerical_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])
```

- **Numerical features** (`Age`, `total_spent`): scaled and mean-imputed.
- **Categorical features** (`HomePlanet`, `CryoSleep`, `Destination`, `VIP`, `Deck`, `Side`): mode-imputed, one-hot encoded, and scaled (`with_mean=False` to preserve sparsity from one-hot encoding).

---

## 🤖 Modeling & Evaluation

Three classifiers were trained as an end-to-end pipeline (`preprocessing + model`) on a 75/25 train/validation split (`random_state=0`):

| Model | Configuration |
|-------|----------------|
| K-Nearest Neighbors | `n_neighbors=15`, `metric='minkowski'`, `p=2` (Euclidean) |
| Decision Tree | `criterion='entropy'`, `random_state=0` |
| Random Forest | `n_estimators=10`, `criterion='entropy'`, `random_state=0` |

Each model was evaluated using **accuracy** and a **confusion matrix** on the held-out validation split.

---

## 📈 Results

| Model | Accuracy | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|-------|----------|-------------------------------------------|
| **K-Nearest Neighbors** | **0.740** | `[[890, 187], [378, 719]]` |
| Decision Tree | 0.677 | `[[715, 362], [341, 756]]` |
| Random Forest | 0.706 | `[[796, 281], [359, 738]]` |

**K-Nearest Neighbors achieved the best validation accuracy (74.0%)** and was selected as the final model for generating test set predictions (`save.csv`).

---

## 🧭 Critical Analysis & Next Steps

As a baseline iteration, this project establishes a clean, leak-free preprocessing pipeline and a reasonable feature set — but there is clear room to improve beyond ~74% accuracy (top leaderboard solutions for this dataset typically reach 0.80+). Honest next steps for a v2 iteration:

- **Unpack `PassengerId` further:** The group ID (`gggg`) was discarded along with `PassengerId`, but group membership is likely informative (families/groups may share `Transported` outcomes). Re-introducing a `GroupSize` or `IsAlone` feature could add signal.
- **Re-evaluate dropping `Num`:** Cabin number was dropped purely on cardinality grounds; binning it (e.g., into cabin-number ranges) might recover useful spatial information instead of discarding it entirely.
- **Decision Tree/Random Forest underperforming KNN** is unusual and suggests these models are under-tuned — `n_estimators=10` for Random Forest is very low (typically 100–500), and neither model used hyperparameter search.
- **No cross-validation:** Results are based on a single train/validation split. `cross_val_score` or `StratifiedKFold` would give a more robust accuracy estimate and reduce variance from the split itself.
- **No hyperparameter tuning:** `GridSearchCV`/`RandomizedSearchCV` over KNN's `n_neighbors`, tree depth/min-samples for the Decision Tree, and `n_estimators`/`max_depth` for Random Forest would likely close much of the gap to top leaderboard scores.
- **Untried model families:** Gradient boosting methods (XGBoost, LightGBM, CatBoost) tend to perform strongly on this type of tabular, mixed-type dataset and would be a natural next experiment.
- **Spending feature granularity:** Collapsing all five amenity columns into `total_spent` simplifies the feature space but may discard useful patterns (e.g., CryoSleep passengers should have ~$0 across all amenities — a mismatch could itself be a signal).

---

## 📁 Project Structure

```
spaceship-titanic/
│
├── spaceship-titanic.ipynb   # Main notebook: EDA, feature engineering, modeling
├── save.csv                  # Generated submission file (PassengerId, Transported)
└── README.md
```

---

## ▶️ How to Run

1. Download the dataset from the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic/data) and place `train.csv` and `test.csv` in an `input/spaceship-titanic/` directory (or update the file paths in the notebook).
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the notebook `spaceship-titanic.ipynb` end-to-end. This will produce `save.csv`, formatted for direct submission to the Kaggle leaderboard.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (`Pipeline`, `ColumnTransformer`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`)
- **Evaluation:** Accuracy score, confusion matrix

---

## 👤 Author

**[Your Name]**
📧 [your.email@example.com] | 🔗 [LinkedIn](https://linkedin.com/in/your-profile) | 💼 [Portfolio](https://your-portfolio.com)

---

⭐ If you found this project useful or interesting, consider giving it a star!
