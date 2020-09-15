from django.contrib import admin
from Movies.models import Movie,MovieLab,RoleTable,MovieType
from Users.models import User
# Register your models here.

class MovieAdmin(admin.ModelAdmin):
    list_display = ('movie_id','movie_name','movie_length','movie_releaseTime','movie_intro','movie_cover')

admin.site.register(Movie,MovieAdmin)
admin.site.register(MovieLab)
admin.site.register(RoleTable)
admin.site.register(MovieType)
#admin.site.register(User)