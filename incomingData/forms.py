from django import forms
#from django.core.context_processors import csrf

class InformationFromHomepage(forms.Form):
    your_name=forms.CharField(label='Your name', max_length=100)
    email=forms.EmailField(label='Email Address')
    text_to_analyze=forms.CharField(label='Text to Analyze',max_length=280)
