from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.
class Celebrity(models.Model):#明星信息表
    genderType = (('male', "男"),('female', "女"),)#定义性别选项

    celebrity_id = models.AutoField(primary_key = True, verbose_name = "明星id")
    celebrity_name = models.CharField(max_length = 50, verbose_name = "明星名字")
    celebrity_gender = models.CharField(max_length=10, choices=genderType, default="男", null=True, blank=True, verbose_name='明星性别')
    # celebrity_cover = models.ImageField(upload_to='CelebrityCover/%Y%m%d', null=True, verbose_name='明星封面图片路径')
    celebrity_cover = models.CharField(max_length=150, null=True, blank=True, verbose_name='明星封面图片路径')
    celebrity_doubanID = models.IntegerField(null=True, blank=True, verbose_name='豆瓣id')
    # celebrity_birthyear = models.CharField(max_length=4,verbose_name='出生日期',null=True)
    # celebrity_intro = models.TextField(max_length=500,verbose_name='明星简介',null=True)

    def __str__(self):
        return self.celebrity_name
