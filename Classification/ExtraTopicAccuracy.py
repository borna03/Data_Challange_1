import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')


def load_classification_data():
    # Clean text file
    open('Extra Topic Backup Results.txt', 'w').close()

    # Write something in text file
    with open('Extra Topic Backup Results.txt', 'a') as f:
        print(f'Amount of total data per category:', file=f)

    # Available topics
    classification_topic = ["BaggageAndSecurity", "FlightExperience", "CustomerService"]
    all_data_list = []

    for topic in classification_topic:
        # Get the data in the csv files as a pandas DataFrame
        csv_file = f'ClassificationData/{topic}.csv'
        data = pd.read_csv(csv_file, delimiter=';', header=None, names=['text', 'topic'])

        # Write total amount of data point per category in file
        with open('Extra Topic Backup Results.txt', 'a') as f:
            print(f'{topic}: {data.shape[0]}', file=f)

        # Append all data to a list
        all_data_list.append(data)
        print(f"Loaded data from {csv_file}")
    # Concatenate all data to a DataFrame
    combined_data = pd.concat(all_data_list, ignore_index=True)

    return combined_data


def preprocess_data(combined_data):
    # Remove NaN values
    combined_data['text'].fillna('', inplace=True)

    # Put sentences and labels into a list
    sentences = combined_data['text'].tolist()
    labels = combined_data['topic'].tolist()

    # Preprocessing for Vectorization (lowercase, lemmatizing, removing stopwords)
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in range(len(combined_data)):
        # print(combined_data['text'][i])
        review = re.sub('[^A-Za-z]', ' ', combined_data['text'][i])
        review = review.lower()
        review = review.split()  # get list of words
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    return sentences, labels, corpus


def train_classifier(corpus, combined_data):
    # Initialize TfidfVectorizer
    tfidf = TfidfVectorizer(ngram_range=(2, 2), max_features=20000)
    X_tf = tfidf.fit_transform(corpus).toarray()

    # Split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X_tf, combined_data['topic'], test_size=0.25, random_state=0)

    # Initialize Logistic Regression Model
    lr_classifier = LogisticRegression()
    lr_classifier.fit(X_train, y_train)

    return tfidf, lr_classifier, y_test, X_test


def classify_tweets(lr_classifier, y_test, X_test, sentences):
    # Predict topic
    lr_y_pred = lr_classifier.predict(X_test)

    # Confusion matrix
    confusion_mat = confusion_matrix(y_test, lr_y_pred)
    print(confusion_mat)

    # Accuracy test
    accuracy = accuracy_score(y_test, lr_y_pred)
    print(f'Accuracy of Logistic Regression on TfIdf Vectorizer data {accuracy * 100}%')

    # Put sentence, actual label and predicted label into a DataFrame
    df_topic_pred = pd.DataFrame(list(zip(sentences, y_test, lr_y_pred)),
                                 columns=['Text', 'Actual Topic', 'Predicted Topic'])

    # Write confusion matrix, accuracy score and DataFrame into a text tile
    with open('Extra Topic Backup Results.txt', 'a') as f:
        print('', file=f)
        print(f'{confusion_mat}', file=f)
        print('', file=f)
        print(f'Accuracy of Logistic Regression on TfIdf Vectorizer data: {accuracy * 100}%', file=f)
        print('', file=f)
        print(f'{df_topic_pred.to_string()}', file=f)

    return df_topic_pred


def run():
    combined_data = load_classification_data()
    print(combined_data)
    sentences, labels, corpus = preprocess_data(combined_data)
    print(sentences)
    print(labels)
    print(corpus)
    tfidf, lr_classifier, y_test, X_test = train_classifier(corpus, combined_data)
    df_topic_pred = classify_tweets(lr_classifier, y_test, X_test, sentences)
    print(df_topic_pred.to_string())


run()
