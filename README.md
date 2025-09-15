# AI Business Coach

**Tech Stack:** SQLite, Jupyter, Python, PyTorch, MLOps, TypeScript, Docker, React, Node.js

---

## Author
Phillip Harris

---

## Synopsis
Empower business analysts and owners to proactively combat customer churn with AI-driven, actionable intelligence. **AI Business Coach** is a bespoke web application featuring a specialized AI chatbot at its core, helping users gain real-time insights into customer behavior and retention strategies with AI-powered guidance.

The system is powered by a robust, object-oriented churn prediction pipeline trained on a real-world Telco dataset. It implements a three-stage cascade model combining Random Forest, ANN, and Gradient Boosting classifiers. This innovative design allows the system to identify complex, "edge-case" churners with superior recall and precision—a critical advantage in competitive markets. Architected to be completely free of data leakage, the pipeline showcases a best-practices approach to MLOps.

This project is a comprehensive portfolio piece demonstrating the seamless integration of full-stack development (Node.js, React, TypeScript, Docker) with advanced data science and machine learning expertise (Python, PyTorch, Jupyter, SQLite). It highlights the creation of a sophisticated, production-ready system where basic AI approaches would struggle, providing not just data, but actionable strategic insights to drive growth.

---

## Clone or Download
git clone https://github.com/HKtrill/ai-business-coach.git
cd ai-business-coach
npm install      # or yarn

## 📁 Project Structure (Initial)

```plaintext
ai-business-coach/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── test_splits/
├── churn_pipeline/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── leakage_monitor.py
│   ├── cascade_model.py
│   └── experiment_runner.py
├── chatbot_pipeline/
│   ├── __init__.py
│   ├── user_input_handler.py
│   ├── query_processor.py
│   ├── churn_prediction_interface.py
│   └── response_generator.py
├── utils/
│   └── utils.py
├── notebooks/
│   └── churn_pipeline.ipynb
├── BasePipeline.py
└── README.md

```
## 📦 Installation & Setup

*(To be updated during development)*

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd ../frontend
npm install
npm start
```
