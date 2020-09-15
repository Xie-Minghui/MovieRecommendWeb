from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.
from Users.models import User
from Movies.models import Movie
class Rating(models.Model):
    ratingID = models.AutoField(primary_key = True,verbose_name = "rating id")
    userID = models.ForeignKey(User,on_delete=models.CASCADE)
    movieID = models.ForeignKey(Movie,on_delete=models.CASCADE)
    rating = models.FloatField(default = 0.0,verbose_name = "评分")
    timestamp = models.IntegerField(null=True, blank=True, default=0,verbose_name="时间戳")
