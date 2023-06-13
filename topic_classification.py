from data_load import *
import time
import tweetnlp
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def cutoff_value(probabilities):
    topic = None
    for key, value in probabilities.items():
        if (value > 0.5) and (key == max(probabilities, key=probabilities.get)):  # cutoff value for uncertainty
            topic = key
            break
        else:
            topic = 'Uncertain'
    return topic


def topic_single_sentiment_to_airline(id_str):
    """
    Topic Classification and subsequent sentiment analysis per topic.
    Using TweetNLP single-label model for topic classification (See https://github.com/cardiffnlp/tweetnlp).
    Using roBERTa model for sentiment analysis (see https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).
    For all tweets for which the airline is mentioned (@) or is replied to.
    Not including re-tweets
    :param id_str: id of airline
    """
    start_time = time.time()

    # Setup Topic Classification Model
    print('Loading topic classification model')
    total_count = 0
    topic_count_before = dict()
    topic_count_after = dict()
    stop_count = 0
    # topic_model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-topic-multi", max_length=128)
    topic_model = tweetnlp.load_model('topic_classification', multi_label=False)
    print('Loading topic classification model successful')

    # Setup Sentiment Analysis Model
    print('Loading sentiment analysis model')
    sent_MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(sent_MODEL)
    config = AutoConfig.from_pretrained(sent_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sent_MODEL)
    sentiment_model.save_pretrained(sent_MODEL)
    tokenizer.save_pretrained(sent_MODEL)
    pos_sent = 0
    neu_sent = 0
    neg_sent = 0
    unc_sent = 0
    print('Loading sentiment analysis model successful')

    error_count = 0

    # Find all tweets for which the airline is mentioned (@) or is replied to:
    query = {'$or': [{'in_reply_to_user_id_str': id_str}, {'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}}]}

    print(f'Expected amount of iterations/tweets: {collection.count_documents(query)}')

    all_tweets = collection.find(query)
    for doc in tqdm(all_tweets):
        try:
            # Only evaluate the first x amount of tweets that are evaluated
            stop_count += 1
            total_count += 1
            text = doc['text']
            processed_text = preprocess(text)

            # Predict topic
            # topic_dict = model.predict(f'{processed_text}', return_probability=True)
            topic_dict = topic_model.topic(f'{processed_text}', return_probability=True)
            topic = topic_dict['label']

            # Predict sentiment
            encoded_input = tokenizer(processed_text, return_tensors='pt')
            output = sentiment_model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            score_pos = scores[2]
            score_neu = scores[1]
            score_neg = scores[0]
            if (score_pos > score_neu) and (score_pos > score_neg) and (score_pos > 0.5):
                label = 'positive'
                pos_sent += 1
            elif (score_neu > score_pos) and (score_neu > score_neg) and (score_neu > 0.5):
                label = 'neutral'
                neu_sent += 1
            elif (score_neg > score_pos) and (score_neg > score_neu) and (score_neg > 0.5):
                label = 'negative'
                neg_sent += 1
            else:
                label = 'uncertain'
                unc_sent += 1

            # Put sentiment label as value of 'sentiment' key
            newvalues_sent = {'$set': {'sentiment': label}}
            collection.update_one(doc, newvalues_sent)

            # Counter - Nested dictionary with topics and sentiments (no topic cutoff)
            if topic in topic_count_before:
                topic_count_before[topic][label] += 1
            else:
                sentiment_count_before: dict = {'positive': 0, 'negative': 0, 'neutral': 0, 'uncertain': 0}
                sentiment_count_before[label] += 1
                topic_count_before[topic] = sentiment_count_before

            probabilities = topic_dict['probability']
            # Implement cutoff value for topics
            topic = cutoff_value(probabilities)

            # Put topic classification as value of 'topic' key
            newvalues_topic = {'$set': {'topic': topic}}
            collection.update_one(doc, newvalues_topic)

            # Counter - Nested dictionary with topics and sentiments (with topic cutoff)
            if topic in topic_count_after:
                topic_count_after[topic][label] += 1
            else:
                sentiment_count_after: dict = {'positive': 0, 'negative': 0, 'neutral': 0, 'uncertain': 0}
                sentiment_count_after[label] += 1
                topic_count_after[topic] = sentiment_count_after

        # There was a single error in the middle of the run somewhere, so print error and continue the loop
        except Exception as e:
            print(e)
            error_count += 1

    print(f'Error count: {error_count}')

    # Convert counters to nested DataFrames
    df_before = pd.DataFrame.from_dict(topic_count_before, orient='index')
    df_after = pd.DataFrame.from_dict(topic_count_after, orient='index')

    print(f'Total evaluated tweets: {total_count}')
    print('\n')

    print(topic_count_before)
    print (topic_count_after)
    print('\n')

    print(f'Topic occurrences & Sentiment (before cutoff): \n {df_before}')
    print(f'Topic occurrences & Sentiment (after cutoff): \n {df_after}')

    print('\n')
    print(f'Total positive sentiments: {pos_sent}')
    print(f'Total neutral sentiments: {neu_sent}')
    print(f'Total negative sentiments: {neg_sent}')
    print(f'Total uncertain sentiments: {unc_sent}')

    print('\n')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)

    with open('zzAnalysisLogs.txt', 'a') as f:
        print(f'Airline (TO) id: {id_str}', file=f)
        print('\n')

        print(f'Error count: {error_count}', file=f)
        print('\n')

        print(f'Total lines: {total_count}', file=f)
        print(topic_count_after, file=f)
        print('\n')

        print("Time: ", elapsed_time, file=f)

def topic_single_sentiment_from_airline(id_str):
    """
    Topic Classification and subsequent sentiment analysis per topic.
    Using TweetNLP single-label model for topic classification (See https://github.com/cardiffnlp/tweetnlp).
    Using roBERTa model for sentiment analysis (see https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).
    For all tweets for which the airline is the one tweeting.
    Not including re-tweets
    :param id_str: id of airline
    """
    start_time = time.time()

    # Setup Topic Classification Model
    print('Loading topic classification model')
    total_count = 0
    topic_count_before = dict()
    topic_count_after = dict()
    stop_count = 0
    # topic_model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-topic-multi", max_length=128)
    topic_model = tweetnlp.load_model('topic_classification', multi_label=False)
    print('Loading topic classification model successful')

    # Setup Sentiment Analysis Model
    print('Loading sentiment analysis model')
    sent_MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(sent_MODEL)
    config = AutoConfig.from_pretrained(sent_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sent_MODEL)
    sentiment_model.save_pretrained(sent_MODEL)
    tokenizer.save_pretrained(sent_MODEL)
    pos_sent = 0
    neu_sent = 0
    neg_sent = 0
    unc_sent = 0
    print('Loading sentiment analysis model successful')

    error_count = 0

    # Find all tweets for which the airline is the one tweeting:
    query = {'$or': [{'user.id_str': id_str}]}

    print(f'Expected amount of iterations/tweets: {collection.count_documents(query)}')

    all_tweets = collection.find(query)
    for doc in tqdm(all_tweets):
        try:
            # Only evaluate the first x amount of tweets that are evaluated
            stop_count += 1
            total_count += 1
            text = doc['text']
            processed_text = preprocess(text)

            # Predict topic
            # topic_dict = model.predict(f'{processed_text}', return_probability=True)
            topic_dict = topic_model.topic(f'{processed_text}', return_probability=True)
            topic = topic_dict['label']

            # Predict sentiment
            encoded_input = tokenizer(processed_text, return_tensors='pt')
            output = sentiment_model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            score_pos = scores[2]
            score_neu = scores[1]
            score_neg = scores[0]
            if (score_pos > score_neu) and (score_pos > score_neg) and (score_pos > 0.5):
                label = 'positive'
                pos_sent += 1
            elif (score_neu > score_pos) and (score_neu > score_neg) and (score_neu > 0.5):
                label = 'neutral'
                neu_sent += 1
            elif (score_neg > score_pos) and (score_neg > score_neu) and (score_neg > 0.5):
                label = 'negative'
                neg_sent += 1
            else:
                label = 'uncertain'
                unc_sent += 1

            # Put sentiment label as value of 'sentiment' key
            newvalues_sent = {'$set': {'sentiment': label}}
            collection.update_one(doc, newvalues_sent)

            # Counter - Nested dictionary with topics and sentiments (no topic cutoff)
            if topic in topic_count_before:
                topic_count_before[topic][label] += 1
            else:
                sentiment_count_before: dict = {'positive': 0, 'negative': 0, 'neutral': 0, 'uncertain': 0}
                sentiment_count_before[label] += 1
                topic_count_before[topic] = sentiment_count_before

            probabilities = topic_dict['probability']
            # Implement cutoff value for topics
            topic = cutoff_value(probabilities)

            # Put topic classification as value of 'topic' key
            newvalues_topic = {'$set': {'topic': topic}}
            collection.update_one(doc, newvalues_topic)

            # Counter - Nested dictionary with topics and sentiments (with topic cutoff)
            if topic in topic_count_after:
                topic_count_after[topic][label] += 1
            else:
                sentiment_count_after: dict = {'positive': 0, 'negative': 0, 'neutral': 0, 'uncertain': 0}
                sentiment_count_after[label] += 1
                topic_count_after[topic] = sentiment_count_after

            # There was a single error in the middle of the run somewhere, so print error and continue the loop
        except Exception as e:
            print(e)
            error_count += 1

    print(f'Error count: {error_count}')

    # Convert counters to nested DataFrames
    df_before = pd.DataFrame.from_dict(topic_count_before, orient='index')
    df_after = pd.DataFrame.from_dict(topic_count_after, orient='index')

    print(f'Total evaluated tweets: {total_count}')
    print('\n')

    print(topic_count_before)
    print (topic_count_after)
    print('\n')

    print(f'Topic occurrences & Sentiment (before cutoff): \n {df_before}')
    print(f'Topic occurrences & Sentiment (after cutoff): \n {df_after}')

    print('\n')
    print(f'Total positive sentiments: {pos_sent}')
    print(f'Total neutral sentiments: {neu_sent}')
    print(f'Total negative sentiments: {neg_sent}')
    print(f'Total uncertain sentiments: {unc_sent}')

    print('\n')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)

    with open('zzAnalysisLogs.txt', 'a') as f:
        print(f'Airline (FROM) id: {id_str}', file=f)
        print('\n')

        print(f'Error count: {error_count}', file=f)
        print('\n')

        print(f'Total lines: {total_count}', file=f)
        print(topic_count_after, file=f)
        print('\n')

        print("Time: ", elapsed_time, file=f)
