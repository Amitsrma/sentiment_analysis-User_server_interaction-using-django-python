# sentiment_analysis-User_server_interaction-using-django-python
This is first stages of study to model user-server interaction and employ neural network to analyze sentiment of sentence.
a. If you have python installed in your system, you can go to step 1 below, if you have not, install python. Using anaconda would be easier and it can be downloaded from following link: 
https://www.anaconda.com/distribution/#download-section 
select appropriate operating system and download

1. open command prompt/Anaconda prompt (anaconda prompt will be available in start button, under anaconda folder)
	to open command prompt, press windows+R button, type cmd and press enter
2. create a virtual environment, use following command for creating it: 
	
	pip install virtualenv <---- installs virtual environment
	
	virtualenv myenv <---- creates virtual environment of name myenv
3. activate virtual environment
	
	in the prompt type following:
	
	\myenv\Scripts\activate
	
	and press enter.
4. find requirements.txt in project folder. Go to the location in command prompt/anaconda prompt and use following command
	
	pip install -r requirements.txt
		
---> let the requirements be installed. Then there is some additional files to be copied to be done.
	
copy Bag_of_words_model.py, energy_bids.csv, train_and_save_model.py and tweets_grabber.py to following location: 
.\myenv\Lib\site-packages\
5. now use following command in command/Anaconda prompt
	
	python (press enter)
	
	import nltk (press enter)
	
	nltk.download('stopwords')
	
	#downloads stopwords of nltk library
	
	exit()

Now environment is setup for running the server. Go to project folder in command/anaconda prompt and use following command in succession:

I> python manage.py makemigrations incomingData

II> python manage.py migrate

III> python manage.py runserver
