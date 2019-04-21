from django.db import models
import pandas as pd
#from datetime import datetime
from sklearn import tree
import tweets_grabber
import Bag_of_words_model

'''
#these parts were created to check how the information interacts in the system
#if you want to check how to access some of the information models, you can check views.py
# Create your models here.
class Dimension_Details(models.Model):
    weight= models.FloatField(default = 0.0)
    width= models.FloatField(default = 0.0)
    shoesize= models.FloatField(default = 0.0)
    time1= models.DateTimeField('date published')

    def __str__(self):
        return str(self.weight)+' at '+str(self.time1)
    
class PredictiveModel(models.Model):
    weight= models.FloatField(default = 0.0)
    width= models.FloatField(default = 0.0)
    shoesize= models.FloatField(default = 0.0)
    result = models.CharField(max_length = 32, editable=False)
    time1= models.DateTimeField('date published')

    def __str__(self):
        return str(self.weight)+' at '+str(self.time1)
    
    def result_of_dimension(self,data):
        clf = tree.DecisionTreeClassifier()
            
        # [height, weight, shoe_size]
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
             [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female',
             'female', 'female', 'male', 'male']

        clf = clf.fit(X, Y)
        print('*****************************')
        print('PREDICTING')
        print('*****************************')
        return clf.predict([data])
    
    def save(self, *args, **kwargs):
	#this part shows how to save result from computation or, any other means to a field
        self.result = self.result_of_dimension([self.weight, self.width, self.shoesize])[0]
        super(PredictiveModel, self).save(*args, **kwargs)

class waterLevel(models.Model):
	reading1=models.FloatField(default = 0.0)
	time1= models.DateTimeField('''auto_now_add = True, blank = False, 
                             ''''date published')

	def __str__(self):
		return str(self.reading1)+' at '+str(self.time1)


class Result_modeller(models.Model):
    def __init__(self, data):
        self.data = data
        self.model_of_analysis = result_generator.modelGenerator(self)
    
    def model(self):
        return self.model_of_analysis.predict(self.data)
'''

class JsonFileModel(models.Model):
    file = models.CharField(max_length=1000)
    
    def save(self, *args, **kwargs):
        dataframe=pd.read_csv('energy_bids.csv')
        file_to_save=Bag_of_words_model.BagofWordsBinaryClassification(dataframe)
        file_buffer=file_to_save.train_DeepNN_model()
        file_buffer=file_to_save.saveDNNmodel()
        self.file = file_buffer
        super(JsonFileModel, self).save(*args, **kwargs)
    #get the trained model from neural network
    #save it in file above

class result_generator(models.Model):
    def __init__(self):
        pass

    def modelGenerator(self):

        clf = tree.DecisionTreeClassifier()
            
        # [height, weight, shoe_size]
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
             [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
        
        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female',
             'female', 'female', 'male', 'male']

        clf = clf.fit(X, Y)
            
        return clf
'''
first create a class that takes tweet
pass it into a functino within the class which opens the model
model is opened from location in the server
model is obtained by running a file on the computer that returns the model and saves it on the computer
saved model is loaded in the class that takes in the tweet
prediction is done using that saved model

get_tweets has function to grab tweets
we can grab the most recent tweets
or, check if the last tweet analyzed is same as the most recent one and analyse those missed ones.
'''

class EnterTweetDatas(models.Model):
    name=models.CharField(max_length=100, null=True, editable=False)
    email=models.CharField(max_length=100, blank=True, null=True, editable=False)
    Tweet=models.CharField(max_length=280,  blank=True, null=True,editable=False)
    result=models.CharField(max_length=100, blank=True, null=True, editable=False)
    '''
    #for future result storage
    def save(self, *args, **kwargs):
        getmostrecenttweet=tweets_grabber.get_tweets()
        self.Tweet = getmostrecenttweet.get_most_recent_tweet()
        super(EnterTweetDatas, self).save(*args, **kwargs)
    '''

class WebScrapeForTweet(models.Model):
    Tweet=models.CharField(max_length=280, null=True, editable=False)
    result=models.CharField(max_length=100, null=True,  editable=False)
    NumOfTweets=models.IntegerField(default=1)
    
    def save(self, *args, **kwargs):
        getmostrecenttweet=tweets_grabber.get_tweets()
        self.Tweet = getmostrecenttweet.get_most_recent_tweet()
        super(EnterTweetDatas, self).save(*args, **kwargs)