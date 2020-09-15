"""UndergraduateMovieRecommand URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from Movies import views as M_v
from Users import views as U_v


urlpatterns = [
    path('api/user/', include('Users.urls')),
    path('api/movie/', include('Movies.urls')),
    path('control/', admin.site.urls),  # 修改默认后台入口路径为control/
    path('movies/',include('Movies.urls')),
    path('api/getMovies/',M_v.get_movies),
    path('api/getRecommendMovies', M_v.return_recommand_movies_for_detail),  # /api/getRecommendMovies/?movie_id=1 ,返回电影详情页上的推荐结果
    path('api/getMovieDetail/<int:movie_id>',M_v.return_movie_json),
    path('api/getHomepage',U_v.home_page_recommend),
    path('api/login',U_v.login),
    path('api/loginout',U_v.loginout),
    path('api/rating/',U_v.ratMovie),
    path('api/toCollect/',U_v.collectMovie),
    path('api/getSimilarMovies',M_v.return_similar_movies_for_detail),
]
