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
tweeted_at_langs_v = {'en': 149499, 'und': 9853, 'in': 138, 'it': 163, 'ar': 26, 'hi': 101, 'et': 93, 'cy': 24, 'es': 587,
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
sentiment_to_airlines = {'KLM': {'id_str': '56377143', 'positive': 24562, 'negative': 37853, 'neutral': 50549, 'uncertain': 8902},
            'AirFrance': {'id_str': '106062176', 'positive': 8344, 'negative': 12601, 'neutral': 17533, 'uncertain': 3399},
            'British_Airways': {'id_str': '18332190', 'positive': 64809, 'negative': 102426, 'neutral': 127238, 'uncertain': 22977},
            'AmericanAir': {'id_str': '22536055'},
            'Lufthansa': {'id_str': '124476322', 'neutral': 26076, 'negative': 20150, 'positive': 11386, 'uncertain': 4618},
            'AirBerlin': {'id_str': '26223583', 'negative': 55, 'neutral': 167, 'positive': 41, 'uncertain': 15},
            'AirBerlin assist': {'id_str': '2182373406', 'neutral': 1, 'negative': 0, 'positive': 1, 'uncertain': 0},
            'easyJet': {'id_str': '38676903', 'neutral': 82699, 'negative': 71154, 'positive': 30533, 'uncertain': 14601},
            'RyanAir': {'id_str': '1542862735', 'neutral': 76594, 'positive': 26412, 'negative': 60735, 'uncertain': 13580},
            'SingaporeAir': {'id_str': '253340062', 'neutral': 18380, 'uncertain': 2031, 'negative': 6851, 'positive': 7955},
            'Qantas': {'id_str': '218730857', 'neutral': 28639, 'negative': 28858, 'uncertain': 7091, 'positive': 16065},
            'EtihadAirways': {'id_str': '45621423', 'neutral': 11581, 'negative': 5192, 'positive': 6488, 'uncertain': 1844},
            'VirginAtlantic': {'id_str': '20626359', 'positive': 32456, 'negative': 36837, 'neutral': 17398, 'uncertain': 5754}}

sentiment_from_airlines = {'KLM': {'id_str': '56377143', 'neutral': 20595, 'uncertain': 1307, 'negative': 6700, 'positive': 7517},
            'AirFrance': {'id_str': '106062176', 'neutral': 4478, 'positive': 2442, 'uncertain': 792, 'negative': 2349},
            'British_Airways': {'id_str': '18332190', 'neutral': 55384, 'positive': 21754, 'negative': 28924, 'uncertain': 8354},
            'AmericanAir': {'id_str': '22536055', 'positive': 36268, 'neutral': 57374, 'negative': 20574, 'uncertain': 10193},
            'Lufthansa': {'id_str': '124476322', 'neutral': 8628, 'positive': 1632, 'negative': 2963, 'uncertain': 742},
            'AirBerlin': {'id_str': '26223583', 'neutral': 0, 'negative': 0, 'positive': 0, 'uncertain': 0},
            'AirBerlin assist': {'id_str': '2182373406', 'neutral': 0, 'negative': 0, 'positive': 0, 'uncertain': 0},
            'easyJet': {'id_str': '38676903', 'negative': 15546, 'neutral': 27986, 'positive': 9915, 'uncertain': 4537},
            'RyanAir': {'id_str': '1542862735', 'neutral': 17232, 'uncertain': 720, 'positive': 1777, 'negative': 1995},
            'SingaporeAir': {'id_str': '253340062', 'neutral': 10233, 'positive': 2060, 'negative': 631, 'uncertain': 677},
            'Qantas': {'id_str': '218730857', 'neutral': 7751, 'positive': 2824, 'negative': 1026, 'uncertain': 1218},
            'EtihadAirways': {'id_str': '45621423', 'positive': 711, 'neutral': 662, 'negative': 51, 'uncertain': 63},
            'VirginAtlantic': {'id_str': '20626359', 'positive': 7321, 'negative': 9631, 'neutral': 4103, 'uncertain': 1530}}

# Topic Classification + Sentiment Analysis
topic_sentiment_to_Virgin = {'business_&_entrepreneurs': {'positive': 1798, 'negative': 3377, 'neutral': 6505, 'uncertain': 832},
                            'daily_life': {'positive': 24271, 'negative': 11155, 'neutral': 22497, 'uncertain': 3609},
                            'science_&_technology': {'positive': 468, 'negative': 607, 'neutral': 1096, 'uncertain': 149},
                            'Uncertain': {'positive': 313, 'negative': 130, 'neutral': 413, 'uncertain': 43},
                            'arts_&_culture': {'positive': 278, 'negative': 60, 'neutral': 186, 'uncertain': 34},
                            'pop_culture': {'positive': 3928, 'negative': 1317, 'neutral': 4692, 'uncertain': 802},
                            'sports_&_gaming': {'positive': 1400, 'negative': 752, 'neutral': 1448, 'uncertain': 285}}
topic_sentiment_from_Virgin = {'business_&_entrepreneurs': {'positive': 782, 'negative': 1024, 'neutral': 4438, 'uncertain': 495},
                               'daily_life': {'positive': 5842, 'negative': 2897, 'neutral': 4589, 'uncertain': 937},
                               'science_&_technology': {'positive': 95, 'negative': 106, 'neutral': 316, 'uncertain': 50},
                               'Uncertain': {'positive': 34, 'negative': 12, 'neutral': 47, 'uncertain': 13},
                               'arts_&_culture': {'positive': 24, 'negative': 1, 'neutral': 13, 'uncertain': 1},
                               'pop_culture': {'positive': 441, 'negative': 36, 'neutral': 170, 'uncertain': 23},
                               'sports_&_gaming': {'positive': 103, 'negative': 27, 'neutral': 58, 'uncertain': 11}}

topic_sentiment_from_British = {'pop_culture': {'positive': 1265, 'negative': 381, 'neutral': 523, 'uncertain': 131},
                                'business_&_entrepreneurs': {'positive': 2341, 'negative': 5796, 'neutral': 29755, 'uncertain': 2408},
                                'daily_life': {'positive': 17248, 'negative': 21431, 'neutral': 21500, 'uncertain': 5459},
                                'science_&_technology': {'positive': 353, 'negative': 878, 'neutral': 2857, 'uncertain': 240},
                                'sports_&_gaming': {'positive': 406, 'negative': 349, 'neutral': 493, 'uncertain': 75},
                                'Uncertain': {'positive': 81, 'negative': 77, 'neutral': 218, 'uncertain': 30},
                                'arts_&_culture': {'positive': 60, 'negative': 12, 'neutral': 38, 'uncertain': 11}}
topic_sentiment_to_British = {'daily_life': {'positive': 46695, 'negative': 65549, 'neutral': 72687, 'uncertain': 14356},
                              'business_&_entrepreneurs': {'positive': 5237, 'negative': 23223, 'neutral': 32883, 'uncertain': 4172},
                              'science_&_technology': {'positive': 1266, 'negative': 3900, 'neutral': 4226, 'uncertain': 695},
                              'Uncertain': {'positive': 766, 'negative': 700, 'neutral': 1074, 'uncertain': 196},
                              'sports_&_gaming': {'positive': 3136, 'negative': 3824, 'neutral': 3659, 'uncertain': 1114},
                              'pop_culture': {'positive': 6993, 'negative': 5048, 'neutral': 12157, 'uncertain': 2352},
                              'arts_&_culture': {'positive': 716, 'negative': 182, 'neutral': 552, 'uncertain': 92}}



