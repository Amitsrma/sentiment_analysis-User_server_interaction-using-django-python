# Generated by Django 2.1.7 on 2019-04-16 22:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('incomingData', '0013_auto_20190416_1441'),
    ]

    operations = [
        migrations.CreateModel(
            name='JsonFileModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.CharField(max_length=1000)),
            ],
        ),
        migrations.DeleteModel(
            name='ResultOfQuery',
        ),
    ]
