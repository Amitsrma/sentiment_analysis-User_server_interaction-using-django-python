from django.shortcuts import render, HttpResponse
from .models import Dimension_Details, Result_modeller, EnterTweetDatas, JsonFileModel
from .forms import InformationFromHomepage
import tensorflow as tf
import Bag_of_words_model
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
#from django.core.context_processors import csrf

# Create your views here.
'''
global BOW_COLS
BOW_COLS=Bag_of_words_model.BagofWordsBinaryClassification()
MODEL=BOW_COLS.get_model()
BOW_COLS=BOW_COLS.get_vectorized_columns()
'''

def incomingData(request):
	return render(request, 'incomingData/login.html')

def detail(request, Dimension_Details_id):
    response = "the gender of person with given dimension is probably %s \b."
#	return HttpResponse(response % Dimension_Details.objects.get(id=Dimension_Details_id).reading1)
    data = []
    data.append(Dimension_Details.objects.get(id=Dimension_Details_id).weight)
    data.append(Dimension_Details.objects.get(id=Dimension_Details_id).width)
    data.append(Dimension_Details.objects.get(id=Dimension_Details_id).shoesize)
    result = Result_modeller([data])
    return HttpResponse(response % result.model())

def index(request):
	readings_list = Dimension_Details.objects.order_by('-time1')[:5]
	output = ', '.join([(q.weight, q.width, q.shoesize) for q in readings_list])
	return HttpResponse(output)

def homepage_view(request):
#    global BOW_COLS
    if request.method=='POST':
        form=InformationFromHomepage(request.POST)
        dataframe=pd.read_csv('energy_bids.csv')
        BowCols=Bag_of_words_model.BagofWordsBinaryClassification(dataframe)
        BowCols=BowCols.get_vectorized_columns()
        if form.is_valid():
            ps=PorterStemmer()
            cd = form.cleaned_data
            your_name = cd ['your_name']
            emailed=cd['email']
            tweeted=cd['text_to_analyze']
            #get prediction of this input
            df=pd.DataFrame(columns=BowCols)
            processed_Tweet=tweeted.lower()
            #first step to eliminate email addresses from text
            #split text based on 
            processed_Tweet= processed_Tweet.split(' ')
            processed_Tweet= ' '.join(i for i in processed_Tweet if '@' not in i)
            processed_Tweet= ''.join(i for i in processed_Tweet if i not in punctuation and not(i.isdigit()) and i!='\t' and i!='\n')
            processed_Tweet= processed_Tweet.split(' ')
            processed_Tweet= ' '.join(ps.stem(i) for i in processed_Tweet if i not in stopwords.words('english'))
            processed_Tweet= processed_Tweet.split(' ')
            for i in processed_Tweet:
                if i in list(df.columns):
                    df.loc[0,i]=1
            df=df.fillna(0)
            print(df.values)
            if sum((df.values)[0])==0:
                res1="your sentiment doesnot match with the sentiment we are trying to analyze"
            else:
                predictor=JsonFileModel.objects.get(id=1).file
                predictor=tf.keras.models.model_from_json(predictor)
                res1=predictor.predict(df.values)
                if res1<=0:
                    res1="This sentiment does not help our cause"
                else:
                    res1="WAY TO GO BRO! WELCOME TO THE BROTHERHOOD"
            info=EnterTweetDatas(name=your_name, email=emailed,
                                 Tweet=tweeted)
            info.save()
            return render(request, 'incomingData/print.html', {'res1':res1})
    return render(request,'index.html')
