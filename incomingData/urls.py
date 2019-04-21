from django.urls import path
from django.shortcuts import render
from django.http import HttpResponse
from . import views

urlpatterns = [
#	path('', views.index, name = 'index'),
	path('incomingData/', views.incomingData, name = 'incomingData'),
	path('<int:Dimension_Details_id>/', views.detail, name = 'detail'),
]
