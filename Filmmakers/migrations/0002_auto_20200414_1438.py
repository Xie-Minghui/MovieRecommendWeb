# Generated by Django 2.2.3 on 2020-04-14 06:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Filmmakers', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='celebrity',
            name='celebrity_name',
            field=models.CharField(max_length=50, verbose_name='明星名字'),
        ),
    ]