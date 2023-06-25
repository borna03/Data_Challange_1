tweet_count_per_month_us = {'2019-06': 15920, '2019-07': 11795, '2019-08': 16934, '2019-09': 22223,
                            '2019-10': 5835, '2019-11': 20614, '2019-12': 10669, '2020-01': 10764,
                            '2020-02': 13490, '2020-03': 28203}
reply_count_per_month_us = {'2019-06': 1528, '2019-07': 1020, '2019-08': 1102, '2019-09': 1453,
                            '2019-10': 393, '2019-11': 954, '2019-12': 883, '2020-01': 1105,
                            '2020-02': 1084, '2020-03': 2816}
tweet_count_per_month_them = {'2019-06': 80686, '2019-07': 70505, '2019-08': 65457, '2019-09': 49173,
                              '2019-10': 16164, '2019-11': 35908, '2019-12': 30748, '2020-01': 38301,
                              '2020-02': 45067, '2020-03': 103188}
reply_count_per_month_them = {'2019-06': 8419, '2019-07': 8008, '2019-08': 6745, '2019-09': 6397,
                              '2019-10': 2112, '2019-11': 4610, '2019-12': 4299, '2020-01': 4845,
                              '2020-02': 6144, '2020-03': 8699}
tweeted_at_langs_v = {'en': 149499, 'und': 9853, 'in': 138, 'it': 163, 'ar': 26, 'hi': 101, 'et': 93, 'cy': 24,
                      'es': 587,
                      'zh': 2, 'fr': 1384, 'pt': 796, 'eu': 20, 'lt': 25, 'de': 82, 'ko': 28, 'nl': 146, 'pl': 25,
                      'tl': 111, 'tr': 35, 'sl': 8, 'ht': 58, 'ja': 24, 'no': 27, 'fi': 32, 'ro': 9, 'lv': 23, 'ca': 42,
                      'sv': 22, 'da': 15, 'hu': 7, 'ru': 12, 'uk': 2, 'cs': 34, 'is': 62, 'vi': 4, 'iw': 20, 'th': 2,
                      'mr': 1, 'bg': 2}
tweeted_at_langs_b = {'und': 16464, 'en': 526169, 'ko': 2495, 'fr': 1874, 'ja': 3038, 'es': 6518, 'tl': 504, 'tr': 137,
                      'lt': 49, 'in': 624, 'ht': 144, 'ca': 199, 'et': 181, 'it': 567, 'pl': 219, 'no': 86, 'pt': 619,
                      'fa': 18, 'nl': 310, 'ur': 1027, 'de': 476, 'hi': 449, 'ar': 985, 'sv': 62, 'vi': 19, 'iw': 36,
                      'lv': 39, 'cs': 79, 'eu': 62, 'sl': 149, 'ro': 120, 'hu': 35, 'da': 104, 'cy': 83, 'fi': 53,
                      'ps': 2, 'ne': 3, 'bn': 2, 'is': 38, 'ru': 31, 'el': 159, 'ta': 2, 'zh': 18, 'th': 17, 'mr': 11,
                      'si': 1, 'kn': 1}
replied_to_langs_v = {'en': 12958, 'es': 4, 'und': 25, 'nl': 2, 'in': 1, 'et': 1, 'no': 1}
replied_to_langs_b = {'en': 63564, 'es': 40, 'ca': 5, 'hi': 4, 'und': 150, 'pt': 17, 'pl': 3, 'cs': 2, 'ht': 5, 'ro': 4,
                      'it': 9, 'tl': 3, 'fr': 16, 'in': 9, 'et': 5, 'de': 10, 'cy': 1, 'ja': 7, 'is': 1, 'fi': 1,
                      'da': 6, 'sv': 1, 'nl': 1, 'no': 1, 'eu': 1}
tweeted_at_langs_Virgin = {'en': 149499, 'und': 9853, 'rest': 4192}
tweeted_at_langs_British = {'en': 526169, 'und': 16464, 'rest': 21645}
replied_to_langs_Virgin = {'en': 12958, 'und': 25, 'rest': 9}
replied_to_langs_British = {'en': 63564, 'und': 150, 'rest': 152}
nested_pie_Virgin = [[136541, 12958], [9828, 25], [4183, 9]]
nested_pie_British = [[462605, 63564], [16314, 150], [21493, 152]]

# Sentiment Analysis
sentiment_to_airlines = {
    'KLM': {'id_str': '56377143', 'positive': 23420, 'negative': 36700, 'neutral': 48484, 'uncertain': 8565},
    'AirFrance': {'id_str': '106062176', 'positive': 7909, 'negative': 12052, 'neutral': 16788, 'uncertain': 3233},
    'British_Airways': {'id_str': '18332190', 'uncertain': 21508, 'positive': 59142, 'neutral': 119969, 'negative': 97021},
    'AmericanAir': {'id_str': '22536055', 'neutral': 131740, 'negative': 175809, 'positive': 74360, 'uncertain': 35283},
    'Lufthansa': {'id_str': '124476322', 'neutral': 24908, 'negative': 19223, 'positive': 10644, 'uncertain': 4412},
    'AirBerlin': {'id_str': '26223583', 'negative': 51, 'neutral': 132, 'positive': 36, 'uncertain': 14},
    'AirBerlin assist': {'id_str': '2182373406', 'neutral': 1, 'negative': 0, 'positive': 1, 'uncertain': 0},
    'easyJet': {'id_str': '38676903', 'neutral': 79585, 'negative': 68556, 'positive': 29045, 'uncertain': 14075},
    'RyanAir': {'id_str': '1542862735', 'neutral': 72391, 'positive': 24735, 'negative': 58059, 'uncertain': 12822},
    'SingaporeAir': {'id_str': '253340062', 'neutral': 17671, 'uncertain': 1946, 'negative': 6594, 'positive': 7511},
    'Qantas': {'id_str': '218730857', 'neutral': 27071, 'negative': 27347, 'uncertain': 6685, 'positive': 15137},
    'EtihadAirways': {'id_str': '45621423', 'neutral': 10941, 'negative': 4890, 'positive': 6179, 'uncertain': 1722},
    'VirginAtlantic': {'id_str': '20626359', 'positive': 31134, 'negative': 35593, 'neutral': 16639, 'uncertain': 5516}}

sentiment_from_airlines = {
    'KLM': {'id_str': '56377143', 'neutral': 19475, 'uncertain': 1253, 'negative': 6334, 'positive': 7094},
    'AirFrance': {'id_str': '106062176', 'neutral': 4227, 'positive': 2322, 'uncertain': 742, 'negative': 2231},
    'British_Airways': {'id_str': '18332190', 'neutral': 52079, 'uncertain': 7894, 'negative': 27089, 'positive': 19548},
    'AmericanAir': {'id_str': '22536055', 'positive': 33002, 'neutral': 52167, 'negative': 18455, 'uncertain': 9214},
    'Lufthansa': {'id_str': '124476322', 'neutral': 8132, 'positive': 1472, 'negative': 2747, 'uncertain': 690},
    'AirBerlin': {'id_str': '26223583', 'neutral': 0, 'negative': 0, 'positive': 0, 'uncertain': 0},
    'AirBerlin assist': {'id_str': '2182373406', 'neutral': 0, 'negative': 0, 'positive': 0, 'uncertain': 0},
    'easyJet': {'id_str': '38676903', 'negative': 14605, 'neutral': 26171, 'positive': 9131, 'uncertain': 4231},
    'RyanAir': {'id_str': '1542862735', 'neutral': 16121, 'uncertain': 666, 'positive': 1596, 'negative': 1771},
    'SingaporeAir': {'id_str': '253340062', 'neutral': 9785, 'positive': 1913, 'negative': 596, 'uncertain': 623},
    'Qantas': {'id_str': '218730857', 'neutral': 7315, 'positive': 2619, 'negative': 974, 'uncertain': 1155},
    'EtihadAirways': {'id_str': '45621423', 'positive': 650, 'neutral': 608, 'negative': 46, 'uncertain': 60},
    'VirginAtlantic': {'id_str': '20626359', 'positive': 7011, 'negative': 9086, 'neutral': 3883, 'uncertain': 1461}}

# Topic Classification + Sentiment Analysis
topic_sentiment_to_Virgin = {
    'daily_life': {'positive': 23321, 'negative': 10620, 'neutral': 21747, 'uncertain': 3458},
    'business_&_entrepreneurs': {'positive': 1722, 'negative': 3261, 'neutral': 6257, 'uncertain': 796},
    'sports_&_gaming': {'positive': 1337, 'negative': 717, 'neutral': 1398, 'uncertain': 266},
    'pop_culture': {'positive': 3738, 'negative': 1280, 'neutral': 4565, 'uncertain': 783},
    'science_&_technology': {'positive': 448, 'negative': 585, 'neutral': 1042, 'uncertain': 143},
    'Uncertain': {'positive': 299, 'negative': 122, 'neutral': 402, 'uncertain': 40},
    'arts_&_culture': {'positive': 269, 'negative': 54, 'neutral': 182, 'uncertain': 30}}
topic_sentiment_from_Virgin = {
    'business_&_entrepreneurs': {'positive': 757, 'negative': 985, 'neutral': 4220, 'uncertain': 473},
    'daily_life': {'positive': 5582, 'negative': 2733, 'neutral': 4311, 'uncertain': 897},
    'science_&_technology': {'positive': 93, 'negative': 95, 'neutral': 287, 'uncertain': 44},
    'Uncertain': {'positive': 32, 'negative': 12, 'neutral': 40, 'uncertain': 13},
    'arts_&_culture': {'positive': 24, 'negative': 1, 'neutral': 12, 'uncertain': 1},
    'pop_culture': {'positive': 426, 'negative': 32, 'neutral': 162, 'uncertain': 22},
    'sports_&_gaming': {'positive': 97, 'negative': 25, 'neutral': 54, 'uncertain': 11}}

topic_sentiment_from_British = {'daily_life': {'positive': 15535, 'negative': 20005, 'neutral': 20070, 'uncertain': 5136},
                                'business_&_entrepreneurs': {'positive': 2154, 'negative': 5530, 'neutral': 28186, 'uncertain': 2308},
                                'sports_&_gaming': {'positive': 352, 'negative': 311, 'neutral': 439, 'uncertain': 67},
                                'pop_culture': {'positive': 1070, 'negative': 335, 'neutral': 461, 'uncertain': 116},
                                'science_&_technology': {'positive': 314, 'negative': 824, 'neutral': 2682, 'uncertain': 228},
                                'Uncertain': {'positive': 71, 'negative': 73, 'neutral': 207, 'uncertain': 29},
                                'arts_&_culture': {'positive': 52, 'negative': 11, 'neutral': 34, 'uncertain': 10}}
topic_sentiment_to_British = {'daily_life': {'positive': 42697, 'negative': 62012, 'neutral': 68680, 'uncertain': 13421},
                              'business_&_entrepreneurs': {'positive': 4832, 'negative': 22288, 'neutral': 31319, 'uncertain': 3962},
                              'sports_&_gaming': {'positive': 2839, 'negative': 3594, 'neutral': 3389, 'uncertain': 1054},
                              'pop_culture': {'positive': 6287, 'negative': 4661, 'neutral': 11123, 'uncertain': 2162},
                              'science_&_technology': {'positive': 1153, 'negative': 3640, 'neutral': 3953, 'uncertain': 645},
                              'Uncertain': {'positive': 701, 'negative': 659, 'neutral': 1002, 'uncertain': 180},
                              'arts_&_culture': {'positive': 633, 'negative': 167, 'neutral': 503, 'uncertain': 84}}


