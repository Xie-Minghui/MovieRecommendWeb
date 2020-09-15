from django.db import models
from baseModel.BaseModel import BaseModel
from Users.models import User
from Movies.models import Movie
# Create your models here.
class BrowseHistory(models.Model):#存放用户浏览历史
    history_id = models.AutoField(primary_key=True, verbose_name='用户浏览记录id')
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie,on_delete=models.CASCADE)