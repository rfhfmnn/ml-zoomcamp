# Club Atlético Independiente Match Outcome Predictor
*Machine Learning Zoomcamp — Midterm Project*

---

## 1. Problem Description & Context

### 1.1 Background & Context
Football (soccer) is one of the most dynamic and competitive sports in the world. In the Argentine football league, match outcomes are influenced by a wide variety of factors including historical rivalries, opponent strength, tactical trends over the years, and particularly the **home-field advantage** (*ventaja de localía*).

Club Atlético Independiente (CAI), based in Avellaneda, Buenos Aires, is one of Argentina's traditional "Cinco Grandes" (Big Five) clubs with a storied history of national and international championships. However, anticipating how the team will perform in any upcoming fixture against different rivals—especially when playing at home in the *Estadio Libertadores de América* versus playing away as a visitor—is a non-trivial challenge with significant variance.

### 1.2 The Machine Learning Problem
The objective of this project is to build an end-to-end Machine Learning classification model that predicts match outcomes for **Club Atlético Independiente**:
- **Target Variable**: Match result from Independiente's perspective (`Win`, `Draw`, `Loss` or binary `Victory vs Non-Victory`).
- **Input Features**:
  - Opponent team (`visitor_team` or `local_team`).
  - Match venue / condition (`is_home`: playing as local or visitor).
  - Tournament / competition context (`date_name`).
  - Match year and historical temporal trends (`anio`).
  - Historical scoring and head-to-head performance indicators.

The dataset used is `data/results.csv`, which aggregates historical match results across Argentine football tournaments, recording match dates, participating teams, goals scored, and tournament divisions.

---

## 2. How the Solution Will Be Used

### 2.1 Use Case & Target Audience
The trained model is designed to assist **sports analysts, tactical technical staff, and football enthusiasts**:
1. **Matchday Preparation & Risk Assessment**: Before an upcoming fixture, analysts can input the next opponent and venue condition to benchmark the expected win probability against historical baselines.
2. **Opponent Matchup Analysis**: Quantify against which opponents the home advantage plays the most decisive role and where the team historically struggles as a visitor.

### 2.2 Operational Workflow & Deployment Architecture
The solution is deployed as an automated prediction service:

```
+------------------------+        HTTP POST JSON Payload        +-----------------------------+
| Client / Analyst       | -----------------------------------> | REST API (Flask / FastAPI)  |
| (Dashboard, CLI, Web)  |                                      | Containerized with Docker   |
+------------------------+ <----------------------------------- +-----------------------------+
                                   Prediction Response:
                           {"outcome": "Win", "probability": 0.68}
```

1. **Input Payload**: A client application sends a JSON request with upcoming match details:
   ```json
   {
     "opponent": "Racing Club",
     "is_home": true,
     "competition": "Liga Profesional",
     "year": 2026
   }
   ```
2. **Preprocessing & Inference**: The service loads the serialized model and preprocessing pipeline (e.g. Scikit-Learn pipeline / DictVectorizer), transforms the categorical and numerical inputs, and generates predictions.
3. **Output Response**: The API responds with the predicted outcome and associated class probabilities, enabling programmatic integration into dashboards or decision-making tools:
   ```json
   {
     "prediction": "Win",
     "win_probability": 0.68,
     "draw_probability": 0.20,
     "loss_probability": 0.12
   }
   ```

---

## 3. Project Structure

```text
├── data/
│   └── results.csv            # Historical Argentine football match dataset
├── notebook.ipynb             # Exploratory Data Analysis (EDA) and Model Training
├── guia.txt                   # Project roadmap and EDA guidelines
└── README.md                  # Project overview, problem description, and instructions
```

*(Note: Additional deployment scripts such as `train.py`, `predict.py`, `Dockerfile`, and dependency files `Pipfile`/`requirements.txt` will be added as the project progresses).*
