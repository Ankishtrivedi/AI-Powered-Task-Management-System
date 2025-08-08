from utils.model_utils import train_models, load_models, predict_task

# Paths
DATA_PATH = "data/tasks.csv"
CLASSIFIER_PATH = "models/classifier.pkl"
PRIORITY_MODEL_PATH = "models/priority_model.pkl"
TFIDF_PATH = "models/tfidf.pkl"

if __name__ == "__main__":
    # Train models
    train_models(DATA_PATH, CLASSIFIER_PATH, PRIORITY_MODEL_PATH, TFIDF_PATH)

    # Load models
    clf, pm, tfidf = load_models(CLASSIFIER_PATH, PRIORITY_MODEL_PATH, TFIDF_PATH)

    # Test prediction
    task_desc = "Fix security vulnerability in payment API"
    category, priority = predict_task(task_desc, clf, pm, tfidf)
    print(f"Task: {task_desc}")
    print(f"Predicted Category: {category}")
    print(f"Predicted Priority: {priority}")
