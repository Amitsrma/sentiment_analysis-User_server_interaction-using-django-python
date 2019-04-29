# Generated by Django 2.1.7 on 2019-03-28 21:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('incomingData', '0005_auto_20190224_0000'),
    ]

    operations = [
        migrations.CreateModel(
            name='Dimension_Details',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('reading1', models.FloatField(default=0.0)),
                ('reading2', models.FloatField(default=0.0)),
                ('reading3', models.FloatField(default=0.0)),
                ('time1', models.DateTimeField(verbose_name='date published')),
            ],
        ),
        migrations.CreateModel(
            name='Result_modeller',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
        ),
        migrations.DeleteModel(
            name='source1',
        ),
        migrations.AlterField(
            model_name='waterlevel',
            name='time1',
            field=models.DateTimeField(verbose_name='auto_now_add = True, blank = False, \n                             date published'),
        ),
    ]