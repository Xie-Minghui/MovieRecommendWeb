from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.


class Comments(models.Model):
    comment_id = models.AutoField(primary_key=True)     #主键自增长
    user = models.ForeignKey('Users.User', on_delete=models.CASCADE)#外键约束
    movie = models.ForeignKey('Movies.Movie', on_delete=models.CASCADE)
    content = models.TextField(max_length=500, verbose_name="评论内容")
    # agree_number = models.IntegerField(default=0, verbose_name="赞同量")
    # disagree_number = models.IntegerField(default=0, verbose_name="踩")
