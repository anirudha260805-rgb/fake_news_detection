"""Classification of news articles as real or fake with TF-IDF and logistic regression.

Input:
    ``news_dataset.csv`` in the current working directory. The CSV contains ``title``, ``text``, ``date``, and ``label`` columns.
      Labels are expected to use ``1`` for real news and ``0`` for fake news.

Workflow:
    1. Combining each article's title and body into one text feature.
    2. Splitting the data into training and testing sets.
    3. Using ``GridSearchCV`` to tune TF-IDF and logistic-regression parameters.
    4. Evaluating the best model on the held-out test set.
    5. Classification using one example article.

The script prints the best hyperparameters, cross-validation score, test
accuracy, classification report, and the example prediction.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Loading and preparing the source dataset.
news=pd.read_csv("news_dataset.csv")

# print(news.head(10))
# print(news.describe())
# print(news.info())

news["date"] = pd.to_datetime(news["date"], format="mixed",errors="coerce")
news["content"]=news["title"].fillna(" ")+" "+news["text"].fillna(" ")
news["label_name"]=news["label"].map({1:"REAL",0:"FAKE"})


# Keeping the test set separate so it is not used during hyperparameter tuning.
X_train,X_test,Y_train,Y_test=train_test_split(
    news["content"],
    news["label"],
    test_size=0.20,
    random_state=42
)

# The pipeline prevents TF-IDF vocabulary information from leaking between CV folds.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("model", LogisticRegression(max_iter=1000))
])

# providing the paramteters and using GridsearchCV for fine tuning.
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "model__C": [0.1, 1, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
# Selecting the parameter combination with the highest mean CV accuracy.
grid_search.fit(X_train, Y_train)

model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluating only after model selection is complete.
Y_pred = model.predict(X_test)

print("Accuracy score:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:")
print(classification_report(
    Y_test,
    Y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"]
))

news["word_count"] = news["content"].str.split().str.len()
news["character_count"] = news["content"].str.len()
news["sentence_count"] = news["content"].str.count(r"[.!?]")

print(news[["word_count", "character_count", "sentence_count"]].describe())

# Display the label distribution before starting the interactive input loop.
print(news["label"].value_counts())
news["label_name"].value_counts().plot(kind="bar")
plt.title("Fake vs Real News")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")
plt.show()


# Predicting the label for a new, unseen article entered by the user.
while True:
    new_data = input("\nEnter news article (or type 'exit'): ").strip()

    if new_data.lower() == "exit":
        print("Program closed.")
        break

    if not new_data:
        print("Please enter some news text.")
        continue

    result = model.predict([new_data])[0]
    probability = model.predict_proba([new_data])[0]

    if result == 1:
        print("Result: REAL")
        print(f"Confidence: {probability[1] * 100:.2f}%")
    else:
        print("Result: FAKE")
        print(f"Confidence: {probability[0] * 100:.2f}%")
    
    