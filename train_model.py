import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# 3-Class Sentiment Dataset
data = {
    "text": [
        "I absolutely love this product! Highly recommended, amazing experience.",
        "Best purchase ever! High quality and very fast shipping.",
        "Terrible product, completely broken and stopped working instantly.",
        "Worst customer support, absolute waste of time and money.",
        "The product is okay, nothing special but works fine.",
        "Average quality, delivered on time. It is decent.",
        "Superb build, fantastic design, extremely satisfied!",
        "Extremely bad experience, item was defective and dirty.",
        "It is acceptable, fair product for the price point.",
        "Pathetic service, money trapped, do not buy this product."
    ],
    "sentiment": [
        "Positive", "Positive", "Negative", "Negative", "Neutral",
        "Neutral", "Positive", "Negative", "Neutral", "Negative"
    ]
}

df = pd.DataFrame(data)
df['cleaned_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Train Model with Multi-class Logistic Regression
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X, y)

# Save Assets
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ 3-Class Sentiment Model Trained and Saved Successfully!")