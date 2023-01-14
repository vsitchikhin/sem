from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

import re
import string
import random
import requests

def remove_noise(tweet_tokens, stop_words=()):

  cleaned_tokens = []

  for token, tag in pos_tag(tweet_tokens):
    token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    token = re.sub("(@[A-Za-z0-9_]+)", "", token)
    if tag.startswith("NN"):
      pos = 'n'
    elif tag.startswith('VB'):
      pos = 'v'
    else:
      pos = 'a'

    lemmatizer = WordNetLemmatizer()
    token = lemmatizer.lemmatize(token, pos)

    if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
      cleaned_tokens.append(token.lower())
  return cleaned_tokens


def get_all_words(cleaned_tokens_list):
  for tokens in cleaned_tokens_list:
    for token in tokens:
      yield token


def get_tweets_for_model(cleaned_tokens_list):
  for tweet_tokens in cleaned_tokens_list:
    yield dict([token, True] for token in tweet_tokens)


positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")


stop_words = stopwords.words("russian")


positive_tweet_tokens = twitter_samples.tokenized("positive_tweets.json")
negative_tweet_tokens = twitter_samples.tokenized("negative_tweets.json")


positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
  positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
  negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


all_pos_words = get_all_words(positive_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)


positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]


classifier = NaiveBayesClassifier.train(train_data)




result = requests.get('https://vue-with-http-499c5-default-rtdb.firebaseio.com/sentenses_places_pars.json').json()

# print(result)

for item in result:
  for name in result[item]:
    for year in result[item][name]:
      for index in result[item][name][year]:
        custom_tweet = index["sentence"]
        custom_token = remove_noise(word_tokenize(custom_tweet))
        index["rate"] = classifier.classify(dict([token, True] for token in custom_token))
        print(index)

        
          
      

# for item in result:
#   for name in result[item]:
#     # print('--------------------------------------\n')
#     # print('--' + name + '--\n')
#     for year in result[item][name]:
#       rating = 0
#       custom_tweets = []
#       custom_tokens = []
#       result_rating = []
#       result_object = {
#         "year": {
#           "sentense": "",
#           "ton": ""
#         }
#       }
#       # print('--------' + year + '--------')
      
#       custom_tweets = result[item][name][year]
#       # print('Количество упоминаний - ' + str(len(custom_tweets)))
#       for custom_tweet in custom_tweets:
#         custom_tokens.append(remove_noise(word_tokenize(custom_tweet)))
        
#       for custom_token in custom_tokens:
#         # requests.post('https://vue-with-http-499c5-default-rtdb.firebaseio.com/sentences_rating.json', {
#         #   result[item][name][year][custom_token]: classifier.classify(dict([token, True] for token in custom_token)),
#         # })
#         if classifier.classify(dict([token, True] for token in custom_token)) == 'Positive':
#           rating += 1
        
#         for sent in result[item][name][year]:
#           requests.post('https://vue-with-http-499c5-default-rtdb.firebaseio.com/sentences_rating.json', json={
#             str(year): {
#               "sentence": str(sent),
#               "rating": str(classifier.classify(dict([token, True] for token in custom_token))),
#               "name": str(name)
#             }
#           })
          
#       # print('Средний рейтинг - ' + "{:.2f}".format(((rating / len(custom_tweets)) * 100)) + '%')
#       # print('\n\n')
#       # requests.post('https://vue-with-http-499c5-default-rtdb.firebaseio.com/sentences_rating.json', {
#       #   # result[item][name][year].count: str(len(custom_tweets)),
#       #   # result[item][name][year].rating: str("{:.2f}".format(((rating / len(custom_tweets)) * 100)) + '%'),
#       #   # str(year): {
#       #   #   "sentence": custom_tweet
#       #   # }
#       #   })
      