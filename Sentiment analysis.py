import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

def load_csv_line_by_line(file_path):
    dct = {}

    # Load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['negative', 'neutral', 'positive']

    # Read CSV file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for counter, line in enumerate(reader):
            if counter != 0:
                tweet = line[10]
                encoded_tweet = tokenizer(tweet, return_tensors='pt')
                output = model(**encoded_tweet)

                scores = output.logits.detach().numpy()[0]
                scores = softmax(scores)  # Apply softmax to obtain probability distribution

                sentiment_label = labels[np.argmax(scores)]

                dct[counter] = [line[1], tweet, sentiment_label]  # Store relevant data in dictionary


    # Print the processed data
    #for key, value in dct.items():
        #print(key, value)

    # Calculate and print the accuracy
    correct_counter = sum(1 for key, value in dct.items() if value[0] == value[2])
    accuracy = correct_counter / len(dct)
    print(accuracy)

# Specify the file path
file_path = 'test_data/Labled tweets.csv'

# Call the function to load and process the CSV file
load_csv_line_by_line(file_path)
