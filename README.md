🚀 Predictive Maintenance & Remaining Useful Life (RUL) Prediction

An end-to-end Machine Learning system for **predictive maintenance and Remaining Useful Life (RUL) prediction** of aircraft engine components using the **NASA C-MAPSS FD001 dataset**.

The project covers the complete ML lifecycle — from data ingestion and validation to feature engineering, model training, evaluation, prediction API, and Streamlit dashboard deployment.

---

## 📌 Project Overview

Traditional maintenance strategies often depend on fixed maintenance schedules or reactive repairs.

Predictive maintenance uses machine sensor data and Machine Learning to estimate when equipment may require maintenance.

This project predicts the **Remaining Useful Life (RUL)** of an engine in terms of remaining operating cycles.

### Business Objective

The system helps maintenance teams:

- Predict remaining engine life
- Identify machines at higher risk
- Plan maintenance before failure
- Reduce unexpected downtime
- Support condition-based maintenance
- Improve maintenance planning

---

## 🏗️ System Architecture


                    NASA C-MAPSS Dataset
                            │
                            ▼
                   Data Ingestion
                            │
                            ▼
                   Data Validation
                            │
                            ▼
                Feature Engineering
                            │
                            ▼
                  Data Transformation
                            │
                            ▼
                   Model Training
                            │
                            ▼
                  Model Evaluation
                            │
                            ▼
                     Trained Model
                     model.pkl
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
        FastAPI REST API              Streamlit App
             │                             │
             ▼                             ▼
       RUL Prediction                Interactive UI
             │
             ▼
       Risk Classification
🎯 Key Features
📊 NASA C-MAPSS FD001 dataset processing
🔍 Data validation
🧹 Data preprocessing
⚙️ Feature engineering
📈 Rolling statistical features
🤖 Machine Learning based RUL prediction
📊 Model evaluation
🔄 Prediction pipeline
🚀 FastAPI REST API
🖥️ Streamlit dashboard
📦 Model artifact management
📝 Logging and exception handling
🐳 Docker support
📁 Modular project architecture
📂 Project Structure
Predictive-Maintenance-Industrial/
│
├── data/
│   └── raw/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── artifacts/
│   ├── feature_columns.pkl
│   ├── model.pkl
│   ├── model_comparison.csv
│   ├── model_metadata.pkl
│   ├── test_metrics.csv
│   └── test_predictions.csv
│
├── logs/
│
├── notebooks/
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── __init__.py
│   ├── logger.py
│   └── exception.py
│
├── app.py
├── api.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
📊 Dataset

This project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

The FD001 subset contains simulated aircraft engine degradation data collected over multiple operating cycles.

Dataset Files
train_FD001.txt
test_FD001.txt
RUL_FD001.txt

Dataset source:

NASA C-MAPSS Jet Engine Simulated Data
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

⚙️ Technologies Used
Programming
Python
Data Analysis
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
Visualization
Matplotlib
Seaborn
API
FastAPI
Uvicorn
Pydantic
Deployment / Application
Streamlit
Docker
Development Tools
Git
GitHub
Jupyter Notebook
VS Code
🧠 Machine Learning Workflow

The project follows a modular machine learning pipeline.

1. Data Ingestion

Loads the NASA FD001 training and testing datasets.

Raw Dataset
     ↓
Data Ingestion
2. Data Validation

Validates the input data structure and required files.

Dataset
   ↓
Validation
   ↓
Valid Dataset
3. Data Transformation

The raw sensor measurements are transformed into machine-learning features.

Feature engineering includes:

Sensor measurements
Operating settings
Rolling mean
Rolling standard deviation

Example:

sensor_2
sensor_2_rolling_mean
sensor_2_rolling_std

These features help capture degradation trends over time.

4. Model Training

Machine Learning models are trained using the transformed dataset.

The trained model is stored as:

artifacts/model.pkl
5. Model Evaluation

The trained model is evaluated using test data.

Evaluation outputs include:

test_metrics.csv
test_predictions.csv
model_comparison.csv
📈 RUL Prediction

The model predicts the estimated number of remaining operating cycles.

For example:

Predicted RUL = 45 cycles

This means the system estimates approximately 45 operating cycles remaining before the expected end of useful life.

🚨 Risk Classification

The API converts predicted RUL into maintenance risk levels.

RUL	Risk Level	Recommendation
< 20	🔴 HIGH	Schedule maintenance immediately
20–49	🟠 MEDIUM	Plan maintenance soon and monitor
>= 50	🟢 LOW	Continue normal monitoring

🚀 Installation
1. Clone the Repository
git clone https://github.com/RadhikaBhedurkar/Predictive-Maintenance-Industrial.git
cd Predictive-Maintenance-Industrial

3. Create Virtual Environment
Windows
python -m venv venv

Activate:

venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
📥 Add Dataset

Download the NASA C-MAPSS FD001 dataset and place:

train_FD001.txt
test_FD001.txt
RUL_FD001.txt

inside:

data/raw/

The final structure should look like:

data/
└── raw/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
    
🏋️ Train the Model

From the project root directory:

python -m src.pipeline.training_pipeline

The pipeline performs:

Data Ingestion
       ↓
Data Validation
       ↓
Data Transformation
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation

After successful training, the artifacts/ directory will contain the trained model and evaluation results.

🌐 Run FastAPI

Start the REST API:

python -m uvicorn api:app --reload

API will run at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000//docs

🔍 API Endpoints
Health Check
GET /health

Example response:

{
    "status": "ok"
}
RUL Prediction
POST /predict

Example request:

{
    "setting_1": 0.0,
    "setting_2": 0.0,
    "setting_3": 100.0,
    "sensor_2": 642.0,
    "sensor_3": 1589.0,
    "sensor_4": 1400.0,
    "sensor_7": 553.0,
    "sensor_8": 2388.0,
    "sensor_9": 9045.0,
    "sensor_11": 47.0,
    "sensor_12": 522.0,
    "sensor_13": 2388.0,
    "sensor_14": 8130.0,
    "sensor_15": 8.5,
    "sensor_17": 392.0,
    "sensor_20": 39.0,
    "sensor_21": 23.0
}

Example response:

{
    "predicted_RUL_cycles": 45.32,
    "risk_level": "MEDIUM",
    "recommendation": "Plan maintenance soon and monitor the machine."
}

The predicted RUL value will vary depending on the trained model and input data.

🖥️ Run Streamlit Dashboard

Start the dashboard:

streamlit run app.py

Open:

http://localhost:8501

The dashboard provides an interactive interface for predictive maintenance analysis.

🐳 Docker

The project also contains a Dockerfile for containerized deployment.

After installing and starting Docker Desktop:

docker build -t predictive-maintenance .

Run the container:

docker run -p 8000:8000 predictive-maintenance

The API can then be accessed at:

http://localhost:8501

📦 Model Artifacts

Generated artifacts include:

File	Purpose
model.pkl	Trained ML model
feature_columns.pkl	Model feature list
model_metadata.pkl	Model metadata
test_metrics.csv	Model evaluation metrics
test_predictions.csv	Test predictions
model_comparison.csv	Model comparison results

📈 Project Outputs

The system provides:

Machine Learning
RUL prediction
Model evaluation
Prediction results
Feature engineering
Maintenance Intelligence
High-risk machine identification
Medium-risk monitoring
Low-risk normal monitoring
Maintenance recommendations
Application Layer
REST API
Swagger documentation
Streamlit dashboard

🔐 Production Considerations

For a production deployment, the system can be extended with:

Real-time IoT sensor streaming
PostgreSQL / MySQL integration
Cloud deployment
MLflow model tracking
CI/CD pipeline
Docker + Kubernetes
Authentication and authorization
Monitoring and alerting
Model drift detection
Automated model retraining
Grafana / Power BI monitoring
Kafka-based sensor pipelines

📌 Project Highlights
End-to-End ML Project
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Machine Learning
 ↓
Evaluation
 ↓
Prediction Pipeline
 ↓
FastAPI
 ↓
Streamlit
 ↓
Deployment

This project demonstrates practical knowledge of:

Python
Data Science
Machine Learning
Feature Engineering
Predictive Maintenance
Time-Series Sensor Data
Model Deployment
REST APIs
Streamlit
Docker
Software Engineering Practices

👩‍💻 Author

Radhika Bhedurkar

Data Analyst & Data Science Enthusiast

Connect with me
LinkedIn: https://www.linkedin.com/in/radhika-bhedurkar

GitHub: https://github.com/RadhikaBhedurkar


