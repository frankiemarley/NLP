import pandas as pd
import regex as re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data if not already present
download('stopwords')
download('wordnet')

# Load the dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
data = pd.read_csv(url)
print(data.columns)  # Check the column names

# Update column names based on the dataset
url_column = 'url'  # Use the correct column name for URLs
label_column = 'is_spam'  # Use the correct column name for labels

# Create artificial spam samples (for demonstration purposes)
spam_samples = pd.DataFrame({
    url_column: [
        'http://spam.com/offer', 'http://spammy.com/win', 'http://fakeurl.com/buy', 
        'http://malicious.com/deal', 'http://phishing.com/gift'
    ],
    label_column: [1, 1, 1, 1, 1]
})

# Append the artificial spam samples to the dataset
data = pd.concat([data, spam_samples], ignore_index=True)

# Check the class distribution after adding spam samples
print(data[label_column].value_counts())

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

def preprocess_url(url):
    # Segment URLs by punctuation marks
    url = re.sub(r'[^a-zA-Z0-9]', ' ', url)
    # Remove stopwords
    tokens = [word for word in url.split() if word not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

data[url_column] = data[url_column].apply(preprocess_url)

# Generate a word cloud (optional)
text = ' '.join(data[url_column])
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)

# Display the word cloud (optional)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Split the dataset into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    data[url_column], data[label_column], test_size=0.2, random_state=42, stratify=data[label_column])

# Verify the split
print(y_train.value_counts())
print(y_test.value_counts())

# Vectorize the URLs using TF-IDF with reduced max_features
vectorizer = TfidfVectorizer(max_features=2000)  # Reduced number of features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Build an SVM with default parameters
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy with default SVM: {accuracy_score(y_test, y_pred)}")

# Optimize the previous model using Randomized Search
param_distributions = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

random_search = RandomizedSearchCV(SVC(), param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
random_search.fit(X_train_tfidf, y_train)

# Best parameters and accuracy
best_params = random_search.best_params_
best_accuracy = random_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation accuracy: {best_accuracy}")

# Train the optimized model on the entire training set
optimized_model = SVC(**best_params)
optimized_model.fit(X_train_tfidf, y_train)

# Predict and evaluate the optimized model
y_pred_optimized = optimized_model.predict(X_test_tfidf)
print(f"Accuracy with optimized SVM: {accuracy_score(y_test, y_pred_optimized)}")
