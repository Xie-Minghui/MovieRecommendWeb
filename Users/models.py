from django.db import models


# from baseModel.BaseModel import BaseModel
# Create your models here.
# class UsersManager(models.Manager): #用户管理函数
# 	def add_one_account(self,user_name,password,image_path):
# 		user = self.create(user_name = user_name,password = hash(password),image_path = image_path)
# 		return user
#     def delete_one_account(self,user_name,password):
#         try:
#             user = self.get(user_name = user_name,password = password)
#         except self.model.DoesnotExist
#             user = None
#         return user
class User(models.Model):
    genderType = (
        ('male', "男"),
        ('female', "女"),
    )

    user_id = models.IntegerField(primary_key=True, verbose_name="用户id")
    user_name = models.CharField(max_length=20, verbose_name="用户名")
    password = models.CharField(max_length=20, verbose_name="用户密码")
    signature = models.CharField(null=True, blank=True, max_length=20, verbose_name="个性签名")
    email = models.EmailField(null=True, blank=True, verbose_name='用户邮箱')
    # is_actived = models.BooleanField(default=True, verbose_name='激活状态')
    # image_path = models.ImageField(upload_to='UserAvatar/',null=True,verbose_name = "头像图片路径")
    gender = models.CharField(null=True, blank=True, max_length=10, choices=genderType, default="男", verbose_name='性别')
    user_ageSec = models.IntegerField(null=True, blank=True, verbose_name='用户年龄区间')
    user_occupation = models.CharField(max_length=50, null=True, blank=True, verbose_name='用户职业')

    collected_movie = models.ManyToManyField(to='Movies.Movie', related_name='collected_by_user')  # 用户收藏列表
    rated_movie = models.ManyToManyField(to='Movies.Movie', through='Rating.Rating', related_name='rated_by_user')  # 用户评分过的电影中间表
    # recommend_movies = models.CharField(max_length=300,default="[]",verbose_name='推荐的电影列表')#推荐的电影列表的id

    # objects = UsersManager()#用户管理类

    def __str__(self):
        return self.user_name
