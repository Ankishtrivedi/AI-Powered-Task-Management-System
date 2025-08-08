import streamlit as st
from utils.model_utils import load_models, predict_task
from utils.workload_balancer import balance_workload

# Paths
CLASSIFIER_PATH = "models/classifier.pkl"
PRIORITY_MODEL_PATH = "models/priority_model.pkl"
TFIDF_PATH = "models/tfidf.pkl"

# Load models
clf, pm, tfidf = load_models(CLASSIFIER_PATH, PRIORITY_MODEL_PATH, TFIDF_PATH)

st.title("AI-Powered Task Management System")
st.write("Predicts category & priority for tasks and balances workload.")

# Task Prediction Section
st.header("Task Prediction")
task_input = st.text_area("Enter task description:")
if st.button("Predict"):
    if task_input.strip():
        category, priority = predict_task(task_input, clf, pm, tfidf)
        st.success(f"Category: {category}")
        st.success(f"Priority: {priority}")
    else:
        st.warning("Please enter a task description.")

# Workload Balancer Section
st.header("Workload Balancer")
tasks_input = st.text_area("Enter tasks (one per line):")
team_input = st.text_input("Enter team members (comma separated):")
if st.button("Assign Tasks"):
    tasks = [t.strip() for t in tasks_input.split("\n") if t.strip()]
    team = [m.strip() for m in team_input.split(",") if m.strip()]
    if tasks and team:
        assignments = balance_workload(tasks, team)
        st.write(assignments)
    else:
        st.warning("Please enter both tasks and team members.")
