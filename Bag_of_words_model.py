import pandas as pd
import re
import tensorflow as tf
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from tensorflow import keras

try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

class BagofWordsBinaryClassification(object):
    def __init__(self, dataframe, y='responsive'):
        '''
        this class takes in a dataframe with 2 columns
        one of the columns is the textual field
        other is rating associated with that text field
        this class, BagofWordsBinaryClassification, takes in this dataframe and
        provides options for tensorflow training, logistics regression model training
        compare the model
        and choose the best one
        '''
        self.__dataframe=dataframe
        self.__y=y
        self.__cols = [i for i in self.__dataframe.columns if i not in [self.__y,]][0]
        #print(self.__cols)
        self.__vectorized_dataframe=self.__preProcessData()
    
    def getDataFrame(self):
        '''
        returns the dataframe being storing data for classification
        '''
        return self.__dataframe
    
    def __split_data(self):
        '''
        split the initializing dataframe to training and testing data
        '''
        train, test=train_test_split(self.__dataframe, test_size=0.25)
        return train, test

    def __preProcessData(self):
        '''
        get the dataframe, process all the information in dataframe's textual column
        bring uniformity in data, removes email-addresses, punctuation and digits
        from textual data
        '''
        #preprocessing data
        ps = PorterStemmer()
        
        StopWords=stopwords.words('english')

        for indx in self.__dataframe.index:
            #changing all strings to lower case
            #removing punctuation from all the tweets
            text = self.__dataframe.loc[indx,self.__cols]
            #first step to eliminate email addresses from text
            #split text based on 
            text= text.split(' ')
            text= ' '.join(i for i in text if '@' not in i)
            text= ''.join(i for i in text if i not in punctuation and not(i.isdigit()) and i!='\t' and i!='\n')
            text= text.split(' ')
            text= ' '.join(ps.stem(i) for i in text if i not in StopWords)
            self.__dataframe.loc[indx,'email']=text
        #function to convert processed data into vectorized form
        vectorizer = CountVectorizer()
        #make a count vectorizer
        df_vectorized=vectorizer.fit_transform(self.__dataframe[self.__cols])
        #get names of columns/features for the vectorized form
        self.__cols_features=vectorizer.get_feature_names()
        #convert the vectorized matrix to dataframe
        vectorized_dataframe=pd.DataFrame(df_vectorized.toarray(), columns=self.__cols_features)
        #adding rating column
        self.__y_vectorized = 'responsive_rating'
        vectorized_dataframe[self.__y_vectorized]=self.__dataframe[self.__y]
        return vectorized_dataframe
    
    def get_vectorized_dataframe(self):
        '''
        returns the vectorized dataframe
        '''
        return self.__vectorized_dataframe

    def get_featured_columns(self):
        '''
        returns the featured columns of vectorized columns
        '''
        try:
            return self.__cols_features
        except:
            raise ValueError("Error in ")
    
    def train_DeepNN_model(self, n_epochs=5):
        #split data into train and test
        train,test = train_test_split(self.__vectorized_dataframe, test_size=0.25)
        #get the size of input columns
        self.col_size=len(self.__cols_features)
        #seperate the data into X and Y for training. Convert them to numpy matrices
            #this is training set
        self.__x_train=train[self.__cols_features].values
        self.__y_train=train[self.__y_vectorized].values
            #this is testing set
        self.__x_test=test[self.__cols_features].values
        self.__y_test=test[self.__y_vectorized].values
        
        #size of input
        self.col_size=len(self.__cols_features)
        
        #callbacks. If we reach the accuracy specified in the callback, we will stop training
        class myCallback(tf.keras.callbacks.Callback):
          def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.82):
              print("\nReached 82% accuracy so cancelling training!")
              self.model.stop_training = True
        
        #settingup a callback function
        callbacks = myCallback()
        
        #setting up multilayer perceptron
        self.__model= keras.Sequential([
                #first layer has 2048 perceptron with activation relu
                keras.layers.Dense(2048, input_shape=(self.col_size,), activation='relu'),
                #second layer has 1024 perceptron with sigmoid activation
                keras.layers.Dense(1024, input_shape=(self.col_size,), activation='sigmoid'),
                #last layer has one because we only have one output
                keras.layers.Dense(1),
                ])
        
        #compiling model
        #using 'adam' optimizer and binary_crosentropy for loss
        self.__model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        self.__model.fit(self.__x_train, self.__y_train, epochs=5, callbacks=[callbacks,])
    
    def train_LogisticRegressionModel(self):
        '''
        returns logistic regression model of the data
        '''
        #logistic Regression
        self.__lgr = LogisticRegression()
        self.__lgr.fit(self.__x_train,self.__y_train)
    
    def getLogisticRegPrediction(self):
        '''
        returns logistic regression model's prediction
        '''
        return self.__lgr.predict(self.__x_test)
    
    def getLogisticRegAccuracy(self):
        '''
        returns the tuple of (true positive + true negative) to total result
        '''
        count=0
        total=0
        pred=self.__lgr.predict(self.__x_test)
        for i in (pred==self.__y_test):
            if i==True:
                count += 1
                total += 1
        return(count/total)

    def saveDNNmodel(self):
        self.__modelJSON=self.__model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(self.__modelJSON)
        #self.__model.save_weights('model.h5')
        return self.__modelJSON
    
    def get_vectorized_columns(self):
        return self.__cols_features
    
    def get_model(self):
        try:
            return self.__model
        except:
            raise ValueError('Need to train the model first before requesting for model')


def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def email_dataframe(df_email_column, text_column, rating_column):
    #idea for this was that, since a person will normally have one point of view
    #their sentiment to an issue will be inclined to one side or another
    #based on that, if data contains a lot of email addresses, it alone should be able to act as unique feature
    email_df=pd.DataFrame(columns = [text_column, rating_column])
    email_df['rating']=df_email_column[rating_column]
    for i in df_email_column.index:
        email_df.loc[i,text_column]=' '.join(extract_email_addresses(df_text_column.loc[i,text_column]))
    return email_df

'''
def getWeightCode():
    https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
    https://stackoverflow.com/questions/55155759/weights-and-biases-from-hdf5-file
'''