import tweepy

class get_tweets(object):
    
    def __init__(self, access_tokens = "1115011131955871744-UJNFX7syUjuIhuFdwNA5OYxUIgCgH1",
                access_token_secret = "7c89Rc9JnqkaCB6V1x8HfANB23SPaqGn8rIOvuIVb9qF8",
                consumer_key = "sbCyFfrodjSTWXlX7hpmI7uGb",
                consumer_key_secret = "HL091HCVVZ3ZIGstLVOnSpU6cteICVcJFd280PFLkx4ZeXa7ZH",):
        self.access_tokens = access_tokens
        self.access_token_secret = access_token_secret
        self.consumer_key = consumer_key
        self.consumer_key_secret = consumer_key_secret
        
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_key_secret)
        self.auth.set_access_token(self.access_tokens, self.access_token_secret)
        
        self.api = tweepy.API(self.auth,)
    
    def show_tweets(self, num_to_show = 10):
        '''
        shows *args number of tweets if provided. If no argument is provided
        it prints 10 most recent tweets. It prints string and returns nothing
        '''
        for status in tweepy.Cursor(self.api.home_timeline).items(num_to_show):
            if not(status.truncated):
                print(status.text,)
                print('*****************************************')
            else:
                print(self.__getFullText(status))
                print('*****************************************')


    def __getFullText(self, status):
        return(self.api.get_status(status.id,tweet_mode='extended').full_text)

    def get_most_recent_tweet(self):
        '''
        shows *args number of tweets if provided. If no argument is provided
        it prints 10 most recent tweets. It prints string and returns nothing
        '''
        for status in tweepy.Cursor(self.api.home_timeline).items(1):
            if not(status.truncated):
                return(status.text)
            else:
                return(self.__getFullText(status))
    
    def get_n_tweets(self, n=1):
        '''
        n: number of most recent tweets that user wants to take
        returns a list of tweets, first element is the most recent and successive
        are follow order
        '''
        tweets = []
        for status in tweepy.Cursor(self.api.home_timeline).items(n):
            if not(status.truncated):
                tweets.append(status.text)
            else:
                tweets.append(self.__getFullText(status))
        return tweets