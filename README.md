# AI Business Coach

**Tech Stack:** Python, Jupyter, PyTorch, SQL, PostgreSQL, C++, TypeScript, FastAPI, React, Docker

---

## Author
Phillip Harris

---

## Synopsis
AI Business Coach is a bespoke web application providing business analysts and owners with actionable insights and AI-assisted guidance. Users interact with a specialized AI chatbot trained on an open-source Telco dataset to analyze customer churn, retention, and growth strategies using PyTorch. This project showcases full-stack development and data science skills for portfolio purposes and is not affiliated with any telecommunications company.

---

## Clone or Download
git clone https://github.com/HKtrill/ai-business-coach.git
cd ai-business-coach
npm install      # or yarn

## 📁 Project Structure (Initial)

```plaintext
ai-business-coach/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │  └── routes/
│   │   │      └── chat.py
│   │   ├── models/
│   │   │  └── ai_model.py
│   │   └── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   └── components/
│   │       └── ChatInterface.tsx
│   ├── package.json
│   └── Dockerfile
├── data/
│   └── baseline.csv
├── notebooks/
│   └── ai_dev.ipynb
├── README.md
└── docker-compose.yml
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
