# AI-Powered Task Management System

This project classifies and prioritizes tasks using machine learning and assigns them to users based on workload.

## 🔧 Features

- NLP preprocessing of task descriptions
- Task classification (e.g., Bug, Feature, Improvement)
- Priority prediction (Low, Medium, High)
- Automatic task assignment to least-burdened user

## 📁 Project Structure

- `main.py` - Runs everything end-to-end
- `data/tasks.csv` - Task dataset
- `models/` - Saved ML models
- `utils/` - Helper modules for preprocessing, training, and assignment

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python main.py
