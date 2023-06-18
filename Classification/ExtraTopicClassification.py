import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random


def load_tweets_data():
    tweets = [
        "Just had the worst experience with airport security. They mishandled my baggage and were extremely rude.",
        "Shoutout to the amazing customer service team of @airline! They went above and beyond to help me with my reservation change.",
        "The flight attendants on my last trip were fantastic! They were attentive, friendly, and provided great service.",
        "The legroom on this flight is absolutely terrible. My knees are practically touching the seat in front of me.",
        "Had a terrible experience with the airline's customer service. They were unhelpful and showed no empathy.",
        "Lost my luggage during the flight. Hoping the airline can locate and return it as soon as possible.",
        "Kudos to the airline's customer service team for their prompt response and assistance in resolving my issue.",
        "The in-flight entertainment system was top-notch! Kept me entertained throughout the entire flight.",
        "Airport security procedures were efficient and hassle-free. They made me feel safe and secure.",
        "Extremely disappointed with the customer service. They were unresponsive and provided no solution to my problem.",
        "The food served on this flight was delicious! It exceeded my expectations.",
        "The turbulence during the flight was quite intense. The cabin crew did an excellent job of keeping everyone calm.",
        "The airline's customer service was exceptional! They handled my issue with professionalism and care.",
        "Had a smooth experience with baggage handling at the airport. No issues or delays.",
        "The TSA screening process was a nightmare. Long queues and inconsistent procedures.",
        "Received excellent service from the airline's customer support. They were friendly and resolved my query quickly.",
        "The seats on this flight were surprisingly comfortable. I had a pleasant journey.",
        "Witnessed a security breach at the airport. The authorities need to tighten their security measures.",
        "Disappointed with the poor customer service. They were unresponsive and failed to address my concerns.",
        "The cabin crew on this flight were rude and unprofessional. They made the journey unpleasant."
    ]
    random.shuffle(tweets)
    return tweets


def load_classification_data():
    ClassificationTopic = ["BaggageAndSecurity", "FlightExperience", "CustomerService"]
    combined_data = pd.DataFrame()
    for topic in ClassificationTopic:
        csv_file = f'ClassificationData/{topic}.csv'
        try:
            data = pd.read_csv(csv_file, delimiter=';', header=None, names=['text', 'topic'])
            combined_data = pd.concat([combined_data, data])
            print(f"Loaded data from {csv_file}")
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found.")
    return combined_data


def preprocess_data(combined_data):
    combined_data['text'].fillna('', inplace=True)
    sentences = combined_data['text'].tolist()
    labels = combined_data['topic'].tolist()
    return sentences, labels


def train_classifier(sentences, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    classifier = LogisticRegression()
    classifier.fit(X, labels)
    return vectorizer, classifier


def classify_tweets(tweets, vectorizer, classifier):
    rows = []
    for tweet in tweets:
        new_X = vectorizer.transform([tweet])
        predicted_label = classifier.predict(new_X)[0]
        rows.append([tweet, predicted_label])
    return rows


def print_classification_results(rows):
    for tl in rows:
        print(tl)


def main():
    tweets = load_tweets_data()
    combined_data = load_classification_data()
    sentences, labels = preprocess_data(combined_data)
    vectorizer, classifier = train_classifier(sentences, labels)
    rows = classify_tweets(tweets, vectorizer, classifier)
    print_classification_results(rows)


main()
