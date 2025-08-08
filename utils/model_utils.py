import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.preprocessing import clean_text

def train_models(data_path, classifier_path, priority_model_path, tfidf_path):
    """
    Trains task classifier & priority model.
    """
    df = pd.read_csv(data_path)
    df['clean_text'] = df['task_description'].apply(clean_text)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_text'])

    # Train classifier
    classifier = LogisticRegression()
    classifier.fit(X, df['category'])

    # Train priority model
    priority_model = LogisticRegression()
    priority_model.fit(X, df['priority'])

    # Save models
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    with open(priority_model_path, 'wb') as f:
        pickle.dump(priority_model, f)
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf, f)

    print("Models trained and saved.")

def load_models(classifier_path, priority_model_path, tfidf_path):
    """
    Loads trained models and TF-IDF.
    """
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    with open(priority_model_path, 'rb') as f:
        priority_model = pickle.load(f)
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    return classifier, priority_model, tfidf

def predict_task(task_description, classifier, priority_model, tfidf):
    """
    Predicts category & priority for given task description.
    """
    clean = clean_text(task_description)
    X = tfidf.transform([clean])
    category = classifier.predict(X)[0]
    priority = priority_model.predict(X)[0]
    return category, priority
