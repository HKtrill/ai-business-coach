# AI Business Coach

FastAPI, React, TypeScript, PyTorch, PostgreSQL, Docker

---

## Author
Phillip Harris

---

## Synopsis
AI Business Coach is a full-stack web application designed to provide business owners and analysts with actionable insights, recommendations, and AI-assisted guidance. Users can interact with a custom AI chatbot for business advice focused on customer churn, retention, and growth strategies. The system showcases AI-driven business insights while laying the foundation for future enhancements such as file uploads, feature selection, and per-customer analysis.

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
