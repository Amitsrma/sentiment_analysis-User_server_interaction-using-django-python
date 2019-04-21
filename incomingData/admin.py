from django.contrib import admin
from .models import Dimension_Details, waterLevel, PredictiveModel,EnterTweetDatas, JsonFileModel
# Register your models here.
'''
@admin.register(Dimension_Details)
class Dimension_DetailsAdmin(admin.ModelAdmin):
    list_display = ('weight', 'width','shoesize', 'time1')

@admin.register(waterLevel)
class waterlevelAdmin(admin.ModelAdmin):
    list_display = ('reading1', 'time1')


@admin.register(PredictiveModel)
class PredictiveModelAdmin(admin.ModelAdmin):
    list_display = ('weight', 'width','shoesize', 'result', 'time1')
'''
@admin.register(EnterTweetDatas)
class EnterTweetDatasAdmin(admin.ModelAdmin):
    list_display=('name','email', 'Tweet','result')

@admin.register(JsonFileModel)
class JsonFileModelAdmin(admin.ModelAdmin):
    list_display=('file',)