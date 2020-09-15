from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.
class Recommend(models.Model):
    recommend_id = models.AutoField(primary_key = True,verbose_name = "推荐列表的id")
    user_name = models.ForeignKey('Users.User',on_delete = models.CASCADE)#外键约束
    movie_name = models.ForeignKey('Movies.Movie',on_delete = models.CASCADE)