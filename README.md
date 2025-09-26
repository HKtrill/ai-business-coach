# Project ChurnBot: Turning Churn Into Intelligence

**Tech Stack:** SQLite, Jupyter, Python, PyTorch, MLOps, TypeScript, Docker, React, Node.js

---

## Author
Phillip Harris

---

## Synopsis

**Churnbot** transforms telecommunications customer retention from guesswork into precision science through an intelligent AI assistant built specifically for telecom churn patterns. Unlike generic models that try to predict everything, Churnbot's laser focus on telecom-specific behaviors delivers superior accuracy where it matters most.

## Choose Your Experience

- **Terminal Version (Light):** Perfect for telecom analysts and technical teams who want fast, efficient insights through command-line conversation
- **Dashboard Version (Heavy):** Built for telecom executives who need rich visualizations and executive-ready presentations

Both versions understand telecom churn unlike any general-purpose tool—analyzing call patterns, data usage shifts, billing disputes, and service degradation that generic models miss entirely. Your sensitive subscriber data never leaves your network, yet you get telecommunications-specialized AI insights through natural conversation.

## Local-First Philosophy

Download from GitHub and analyze your subscriber base immediately. **No cloud dependencies, no data breaches, no monthly hosting fees.** Prove that focused, domain-specific models outperform one-size-fits-all solutions while maintaining complete data sovereignty.

## Technical Excellence

This project demonstrates advanced machine learning specialization with a three-stage cascade model combining Random Forest, ANN, and Gradient Boosting classifiers, architected to identify complex telecom churn patterns with superior recall and precision. Built with modern full-stack technologies (Node.js, React, TypeScript) integrated with specialized data science expertise (Python, PyTorch), showcasing production-ready MLOps without data leakage.

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
