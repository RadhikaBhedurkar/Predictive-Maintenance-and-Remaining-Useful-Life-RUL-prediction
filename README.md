# Predictive Maintenance & Remaining Useful Life (RUL) — End-to-End ML

Industrial-style end-to-end machine-learning project using NASA C-MAPSS FD001 simulated jet-engine degradation data.

## Architecture

Dataset -> Data Ingestion -> Data Validation -> RUL/Feature Engineering -> Model Training
-> Model Evaluation -> Best Model Artifact -> Prediction Pipeline -> FastAPI / Streamlit
-> Docker-ready deployment

## Dataset

Download NASA C-MAPSS and place these files in `data/raw/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

C-MAPSS is simulated data. Do not describe it as real factory data.

## Windows setup

```powershell
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
venv\Scripts\activate
```

Install:

```powershell
pip install -r requirements.txt
```

## Train end-to-end

```powershell
python -m src.pipeline.training_pipeline
```

This performs:
1. Data ingestion
2. Data validation
3. RUL creation
4. Rolling feature engineering
5. Train/validation split
6. Random Forest / Gradient Boosting / XGBoost comparison
7. Best model selection
8. Test-set evaluation
9. Artifact creation

Artifacts are saved in `artifacts/`.

## Run Streamlit

```powershell
streamlit run app.py
```

## Run FastAPI

```powershell
uvicorn api:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Example POST body is available in the API docs.

## Docker

Build:

```powershell
docker build -t predictive-maintenance .
```

Run:

```powershell
docker run -p 8000:8000 predictive-maintenance
```

## GitHub

```powershell
git init
git add .
git commit -m "Initial predictive maintenance ML project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Resume description

**Predictive Maintenance & Remaining Useful Life Prediction**
- Developed an end-to-end ML pipeline using NASA C-MAPSS simulated engine sensor data to estimate Remaining Useful Life (RUL).
- Implemented data validation, RUL generation, rolling-window feature engineering, model comparison and automated best-model selection.
- Compared Random Forest, Gradient Boosting and XGBoost using MAE, RMSE and R².
- Built a Streamlit dashboard and FastAPI inference service for model predictions.
- Containerized the application with Docker and implemented production-style logging, exception handling and artifact management.
