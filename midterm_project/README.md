# Club Atlético Independiente Match Outcome Predictor
*Machine Learning Zoomcamp — Midterm Project*

---

## 1. Problem Description & Context

### 1.1 Background & Context
Football (soccer) is one of the most dynamic and competitive sports in the world. In the Argentine football league, match outcomes are influenced by a wide variety of factors including historical rivalries, opponent strength, tactical trends over the years, and particularly the **home-field advantage** (*ventaja de localía*).

Club Atlético Independiente (CAI), based in Avellaneda, Buenos Aires, is one of Argentina's traditional "Cinco Grandes" (Big Five) clubs with a storied history of national and international championships. However, anticipating how the team will perform in any upcoming fixture against different rivals—especially when playing at home in the *Estadio Libertadores de América* versus playing away as a visitor—is a non-trivial challenge with significant variance.

### 1.2 The Machine Learning Problem
The objective of this project is to build an end-to-end Machine Learning classification model that predicts match outcomes for **Club Atlético Independiente**:
- **Target Variable**: Match result from Independiente's perspective (`1` = Victory, `0` = Draw or Loss / Non-Victory).
- **Input Features**:
  - Opponent team (`rival`).
  - Match venue / condition (`condicion_independiente`: `Local` or `Visitante`).
  - Match year and historical temporal trends (`anio`).

The dataset used is [`data/results.csv`](data/results.csv), which aggregates historical match results across Argentine football tournaments, recording match dates, participating teams, goals scored, and tournament divisions.

---

## 2. How the Solution Will Be Used

### 2.1 Use Case & Target Audience
The trained model is designed to assist **sports analysts, tactical technical staff, and football enthusiasts**:
1. **Matchday Preparation & Risk Assessment**: Before an upcoming fixture, analysts can input the next opponent and venue condition to benchmark the expected win probability against historical baselines.
2. **Opponent Matchup Analysis**: Quantify against which opponents the home advantage plays the most decisive role and where the team historically struggles as a visitor.

### 2.2 Operational Workflow & Deployment Architecture
The solution is deployed as an automated REST API web service:

```text
+------------------------+        HTTP POST JSON Payload        +-----------------------------+
| Client / Analyst       | -----------------------------------> | REST API (Flask / Gunicorn) |
| (predict_test.py, Web) |                                      | Containerized with Docker   |
+------------------------+ <----------------------------------- +-----------------------------+
                                   Prediction Response:
                    {"prediction": "Victoria", "win_probability": 0.4563}
```

1. **Input Payload**: A client application sends a JSON request with upcoming match details:
   ```json
   {
     "rival": "Banfield",
     "condicion_independiente": "Local",
     "anio": 2026
   }
   ```
2. **Preprocessing & Inference**: The service loads the serialized model and preprocessing pipeline (`DictVectorizer`), transforms the inputs, and calculates class probabilities using the calibrated decision threshold (`0.41`).
3. **Output Response**: The API responds with the outcome prediction and probabilities:
   ```json
   {
     "match": {
       "anio": 2026,
       "condicion_independiente": "Local",
       "rival": "Banfield"
     },
     "win_probability": 0.4563,
     "non_win_probability": 0.5437,
     "threshold_used": 0.41,
     "victory": true,
     "prediction": "Victoria"
   }
   ```

---

## 3. Project Structure

```text
├── data/
│   └── results.csv            # Historical Argentine football match dataset (committed)
├── k8s/
│   ├── deployment.yaml        # Kubernetes Deployment manifest with liveness/readiness probes
│   └── service.yaml           # Kubernetes Service (LoadBalancer) manifest
├── notebook.ipynb             # Full Exploratory Data Analysis (EDA) and Model Training
├── train.py                   # Standalone training script to train and serialize the final model
├── predict.py                 # Flask REST API serving predictions
├── predict_test.py            # Test script verifying the prediction API with sample matchups
├── model.bin                  # Serialized DictVectorizer and trained Logistic Regression model
├── requirements.txt           # Python package dependencies
├── Pipfile                    # Pipenv dependency specification
├── Pipfile.lock               # Pipenv pinned dependency lockfile
├── Dockerfile                 # Production multi-stage Docker container specification
├── .dockerignore              # Files excluded from Docker container build
├── render.yaml                # Infrastructure-as-code specification for Cloud Deployment (Render)
├── guia.txt                   # Project roadmap and EDA guidelines
└── README.md                  # Comprehensive project documentation
```

---

## 4. Exploratory Data Analysis & Model Training Summary

### 4.1 Exploratory Data Analysis (EDA)
- **Data Quality**: Checked missing values (`df_independiente.isna().sum()`); zero nulls in cleaned subset.
- **Filtering**: Selected matches with `anio <= 2019` to avoid pandemic-era incomplete tournament anomalies.
- **Target Distribution**: Independiente won ~43.9% of matches, with 56.1% resulting in draws or losses.
- **Feature Importance**:
  - **Mutual Information**: `rival` (MI = 0.0278) and `condicion_independiente` (MI = 0.0177) were identified as key predictive factors.
  - **Temporal & Goal Trends**: Analyzed goals scored vs. conceded historically against the league average, confirming stronger win rates at home.

### 4.2 Model Exploration & Hyperparameter Tuning
Four different model architectures were evaluated on a validation set:

| Model Architecture | Hyperparameter Search Space | Best Val ROC-AUC | Best Val Accuracy | Best Val F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | `C` in `[0.001, 0.01, 0.1, 1.0, 10.0]` | **0.6413** | **0.6043** | **0.4819** |
| **Decision Tree** | `max_depth` in `[3, 5, 7, 10]`, `min_samples_leaf` in `[5, 10, 20]` | 0.6193 | 0.5859 | 0.4622 |
| **Random Forest** | `n_estimators` in `[50, 100, 150]`, `max_depth` in `[5, 10]` | 0.6342 | 0.5874 | 0.4139 |
| **XGBoost** | `eta` in `[0.05, 0.1, 0.3]`, `max_depth` in `[3, 6]` | 0.6259 | 0.5828 | 0.4538 |

**Winning Model**: **Logistic Regression** (`C=1.0`) achieved the highest ROC-AUC on validation.
- **Decision Threshold Optimization**: Tuning decision threshold over `[0.3, 0.7]` identified **`0.41`** as the optimal cutoff for maximizing F1-score and Recall.
- **Final Test Performance**: Evaluated on unseen test set: **ROC-AUC: 0.6881**, **Accuracy: 61.50%**, **Recall: 71.43%**, **F1-Score: 0.6144**.

---

## 5. Dependency and Environment Management

You can manage dependencies using either **Pipenv** (recommended) or standard **venv + pip**.

### Option A: Using Pipenv (Recommended)

1. **Install Pipenv** (if not already installed):
   ```bash
   pip install pipenv
   ```
2. **Install dependencies from Pipfile.lock**:
   ```bash
   pipenv install
   ```
3. **Activate the virtual environment**:
   ```bash
   pipenv shell
   ```

### Option B: Using Standard Python Virtual Environment (`venv`)

1. **Create a virtual environment**:
   ```bash
   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate

   # On Linux/macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Reproducibility & Training

### 6.1 Dataset Access
- The dataset is committed directly to this repository at [`data/results.csv`](data/results.csv).
- No external authentication or data download steps are required.

### 6.2 Running the Notebook
- Open [`notebook.ipynb`](notebook.ipynb) in Jupyter and execute all cells.
- All operations are deterministic using fixed seeds (`random_state=42`).

### 6.3 Running the Training Script
To re-train the model and serialize it to [`model.bin`](model.bin):
```bash
python train.py
```

---

## 7. Model Deployment (Local Web Service)

The model is deployed as a REST API using Flask.

### 7.1 Start the Service
```bash
python predict.py
```
*Note: On Windows, `predict.py` automatically uses `waitress` for production WSGI serving; on Linux/Docker it supports `gunicorn`.*

### 7.2 Test the Service
In another terminal window (with the service running), execute:
```bash
python predict_test.py
```

Or test using `curl`:
```bash
curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"rival": "Racing Club", "condicion_independiente": "Local", "anio": 2026}'
```

---

## 8. Containerization with Docker

The service is fully containerized using a lightweight [`Dockerfile`](Dockerfile) based on `python:3.13-slim` and served with `gunicorn`.

### 8.1 Build the Docker Container
```bash
docker build -t independiente-predictor .
```

### 8.2 Run the Docker Container
```bash
docker run -it --rm -p 9696:9696 independiente-predictor
```

### 8.3 Verify Containerized Predictions
While the container is running, execute:
```bash
python predict_test.py
```

---

## 9. Cloud & Kubernetes Deployment

### Option A: Cloud Deployment (Render / Cloud Run)
A [`render.yaml`](render.yaml) Blueprint is included for automated deployment on Render:
1. Connect your GitHub repository to [Render](https://render.com/).
2. Create a new **Web Service** from the repository or use the Blueprint (`render.yaml`).
3. Render automatically builds the `Dockerfile` and deploys the container.
4. Test the live public URL:
   ```bash
   python predict_test.py https://<your-service-name>.onrender.com
   ```

### Option B: Local or Remote Kubernetes Deployment
Declarative Kubernetes manifests are provided under [`k8s/`](k8s/):
- [`k8s/deployment.yaml`](k8s/deployment.yaml): 2 replicas, resource requests/limits, and `/ping` liveness/readiness probes.
- [`k8s/service.yaml`](k8s/service.yaml): Exposes port 80 routing to container port 9696.

**To deploy to a local cluster (e.g. `kind` or `minikube`):**
1. Load the Docker image into the cluster:
   ```bash
   kind load docker-image independiente-predictor:v1
   ```
2. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```
3. Verify the deployment and pods:
   ```bash
   kubectl get pods -l app=independiente-predictor
   kubectl get svc independiente-predictor-service
   ```
4. Forward port or send requests:
   ```bash
   kubectl port-forward svc/independiente-predictor-service 9696:80
   python predict_test.py http://127.0.0.1:9696
   ```
