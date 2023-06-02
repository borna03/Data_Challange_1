from data_load import *
import time
import tweetnlp


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


def topic_single(id_str):
    """
    Topic Classification with single-label model for tweets directed at the airline.
    See https://github.com/cardiffnlp/tweetnlp
    :param id_str: id of airline
    """
    start_time = time.time()
    total_count = 0
    topic_count_before = dict()
    topic_count_after = dict()
    # model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-topic-single", max_length=128)
    model = tweetnlp.load_model('topic_classification', multi_label=False)
    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': id_str}
        }
    }):
        total_count += 1
        text = doc['text']
        processed_text = preprocess(text)
        # topic_dict = model.predict(f'{processed_text}', return_probability=True)
        topic_dict = model.topic(f'{processed_text}', return_probability=True)
        topic = topic_dict['label']

        # Topic occurrence counter (no cutoff)
        if topic in topic_count_before:
            topic_count_before[topic] += 1
        else:
            topic_count_before[topic] = 1

        probabilities = topic_dict['probability']
        cutoff_value(probabilities)

        # Topic occurrence counter (cutoff)
        if topic in topic_count_after:
            topic_count_after[topic] += 1
        else:
            topic_count_after[topic] = 1

        # print(total_count)

    print(f'Total evaluated tweets: {total_count}')
    print(f'Topic occurrences (before): {topic_count_before}')
    print(f'Topic occurrences (after/cutoff): {topic_count_after}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)


def topic_single_stop(id_str, stop):
    """
    Topic Classification with single-label model for tweets directed at the airline.
    With stop number, to check only certain x amount of tweets
    See https://github.com/cardiffnlp/tweetnlp
    :param id_str: id of airline
    :param stop: amount of tweets you want to evaluate
    """
    start_time = time.time()
    total_count = 0
    topic_count_before = dict()
    topic_count_after = dict()
    stop_count = 0
    # model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-topic-multi", max_length=128)
    model = tweetnlp.load_model('topic_classification', multi_label=False)
    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': id_str}
        }
    }):
        # Only evaluate the first x amount of tweets that are evaluated
        if stop_count >= stop:
            break
        else:
            stop_count += 1
            total_count += 1
            text = doc['text']
            processed_text = preprocess(text)
            # topic_dict = model.predict(f'{processed_text}', return_probability=True)
            topic_dict = model.topic(f'{processed_text}', return_probability=True)
            topic = topic_dict['label']

            # Topic occurrence counter (no cutoff)
            if topic in topic_count_before:
                topic_count_before[topic] += 1
            else:
                topic_count_before[topic] = 1

            probabilities = topic_dict['probability']
            cutoff_value(probabilities)

            # Topic occurrence counter (cutoff)
            if topic in topic_count_after:
                topic_count_after[topic] += 1
            else:
                topic_count_after[topic] = 1

            # print(total_count)

    print(f'Total evaluated tweets: {total_count}')
    print(f'Topic occurrences (before): {topic_count_before}')
    print(f'Topic occurrences (after/cutoff): {topic_count_after}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)


def topic_multi(id_str):
    """
    Topic Classification with multi-label model for tweets directed at the airline.
    See https://github.com/cardiffnlp/tweetnlp
    :param id_str: id of airline
    """
    start_time = time.time()
    total_count = 0
    topic_count_before = dict()
    topic_count_after = dict()
    # model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-topic-multi", max_length=128)
    model = tweetnlp.load_model('topic_classification')
    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': id_str}
        }
    }):
        total_count += 1
        text = doc['text']
        processed_text = preprocess(text)
        # topic_dict = model.predict(f'{processed_text}', return_probability=True)
        topic_dict = model.topic(f'{processed_text}', return_probability=True)
        topic_list = topic_dict['label']

        # Topic occurrence counter (no cutoff)
        for topic in topic_list:
            if topic in topic_count_before:
                topic_count_before[topic] += 1
            else:
                topic_count_before[topic] = 1

        probabilities = topic_dict['probability']
        cutoff_value(probabilities)

        # Topic occurrence counter (cutoff)
        if topic in topic_count_after:
            topic_count_after[topic] += 1
        else:
            topic_count_after[topic] = 1

        # print(total_count)

    print(f'Total evaluated tweets: {total_count}')
    print(f'Topic occurrences (before): {topic_count_before}')
    print(f'Topic occurrences (after/cutoff): {topic_count_after}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)


def topic_multi_stop(id_str, stop):
    """
    Topic Classification with multi-label model for tweets directed at the airline.
    With stop number, to check only certain x amount of tweets
    See https://github.com/cardiffnlp/tweetnlp
    :param id_str: id of airline
    :param stop: amount of tweets you want to evaluate
    """
    start_time = time.time()
    total_count = 0
    topic_count_before = dict()
    topic_count_after = dict()
    stop_count = 0
    # model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-topic-multi", max_length=128)
    model = tweetnlp.load_model('topic_classification')
    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': id_str}
        }
    }):
        # Only evaluate the first x amount of tweets that are evaluated
        if stop_count >= stop:
            break
        else:
            stop_count += 1
            total_count += 1
            text = doc['text']
            processed_text = preprocess(text)
            # topic_dict = model.predict(f'{processed_text}', return_probability=True)
            topic_dict = model.topic(f'{processed_text}', return_probability=True)
            topic_list = topic_dict['label']

            # Topic occurrence counter (no cutoff)
            for topic in topic_list:
                if topic in topic_count_before:
                    topic_count_before[topic] += 1
                else:
                    topic_count_before[topic] = 1

            probabilities = topic_dict['probability']
            cutoff_value(probabilities)

            # Topic occurrence counter (cutoff)
            if topic in topic_count_after:
                topic_count_after[topic] += 1
            else:
                topic_count_after[topic] = 1

            # print(total_count)

    print(f'Total evaluated tweets: {total_count}')
    print(f'Topic occurrences (before): {topic_count_before}')
    print(f'Topic occurrences (after/cutoff): {topic_count_after}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)