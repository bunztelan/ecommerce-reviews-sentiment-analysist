import tweepy
from textblob import TextBlob

# Replace with your keys
consumer_key = 'adhioPl4Vsjlb852Spr1P68g6'
consumer_secret = 'mm5GHn72OiHD96wI13ZkxG6n7EGQOAgXMXuvX2utyzmHYWbkiG'
access_token = '1547176329774215169-mtN3ndvyJhHe3HbiGGXX2bD3bKT3u0'
access_token_secret = 'A00uyGVj6GHL4H9QnNmLnC4Iwx49mn0tG0ORVH5bRnK0S'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJRPswEAAAAAymPdue%2FdRqqSSoKJcPyE9n6diEw%3DgHHDjK3iukB4rRO3fh7gmbm92bFEW1wlkCtIvuxxLElQhjeVaZ'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def search_tweets(keyword, number_of_tweets):
    tweets = api.search_tweets(q=keyword, count=number_of_tweets)
    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        print(tweet.text)
        print(analysis.sentiment)

search_tweets('pemilu 2024',5)