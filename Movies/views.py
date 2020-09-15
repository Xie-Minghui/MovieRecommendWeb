from django.shortcuts import render,redirect
from django.http import HttpResponse
from Movies.models import Movie,MovieLab,MovieComment
from Comments.models import Comments
from Filmmakers.models import Celebrity
import os
import json
from Recommend.LFM_sql import LFM, ReadMysql
from Recommend.LFM_test import lfm, sparse_matrix

from Recommend.KNN_test import knn, origin_data
from Recommend.KNN41 import KNN

from Recommend.MR.mrcnn5 import mv_network
from Recommend.MR.mrcnn5_test import mv_net 

from django.http import JsonResponse
import random
#from django.core.serializers.json import json


def return_home_movies(request):
    #按照上映时间抽取最新条目
    movie_objects = Movie.objects.all().order_by('-movie_releaseTime')[:2]
    id_num = 1
    movies = []
    for movie in movie_objects:
        print(movie.movie_name)
        movies.append({
            "id": id_num,
            "title": movie.movie_name,
            "img": "/image/" + str(movie.movie_cover),
            "url": "/movie/" + str(movie.movie_id)
        })
        id_num += 1

    result = {
        "success": True,
        "data": {
            "movies": movies
        }
    }
    
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')

def return_movie_json(request, movie_id):
    movie = Movie.objects.get(movie_id=movie_id)
    comment_objects = MovieComment.objects.filter(movie=movie)
    comment_list = []
    if comment_objects:
        for comment in comment_objects:
            # comment_list.append({"user": comment.user.user_name,"content":comment.content})
            comment_list.append({"user": comment.author_name, "content": comment.content})

    lab_objects = movie.lab.all()  # return all labs objects for this movie
    lab_list = []
    if lab_objects:
        for lab in lab_objects:
            lab_list.append({"type": lab.lab_content, "url": "#"})

    type_objects = movie.types.all()  # return all type objects for this movie
    types = ''
    if type_objects:
        for movie_type in type_objects:
            types = types + movie_type.type_name + ','
        types = types[:-1]

    director_objects = Celebrity.objects.filter(movie=movie, roletable__role='director')
    directors = ''
    for director in director_objects:
        directors = directors + director.celebrity_name + ' / '
    directors = directors[:-3]

    actor_objects = Celebrity.objects.filter(movie=movie, roletable__role='actor')
    actors = ''
    actor_imgs = []
    if actor_objects:
        for actor in actor_objects:
            actors = actors + actor.celebrity_name + ' '
            # actor_imgs.append({"img": "/image/"+str(actor.celebrity_cover)})
            actor_imgs.append({"img": actor.celebrity_cover})
        actors = actors[:-1]

    result = {
        # "id": movie_id,
        "title": movie.movie_name + ' ' + movie.movie_alias,
        "name": movie.movie_name,
        # "poster": "/image/" + str(movie.movie_cover),
        "poster": movie.movie_cover,
        "showtime": movie.movie_releaseTime,  # movie.movie_releaseTime.strftime('%Y-%m-%d')
        "showpos": movie.movie_showPos,
        "length": movie.movie_length,
        "type": types,
        "director": directors,
        "actor": actors,
        "score": movie.movie_grades,
        "introduction": movie.movie_intro,
        "actorimg": actor_imgs,
        "lab": lab_list,
        "comment": comment_list,

    }
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')

def return_recommand_movies_for_detail(request):
    current_movie_id = request.GET.get('id')
    userID = request.session.get('user', None)
    result = {
        "data":[]
    }
    if userID:
        # movie_id_list = RecommendSimilaryItem(int(current_movie_id))
        movie_id_list = RecommendOtherFavoriteMovies(int(current_movie_id))
        for movie_id in movie_id_list:
            movieObj = Movie.objects.get(movie_id=movie_id)

            type_objects = movieObj.types.all()  # return all type objects for this movie
            types = ''
            if type_objects:
                for movie_type in type_objects:
                    types = types + movie_type.type_name + ','
                types = types[:-1]

            result['data'].append({
                "title": movieObj.movie_name,
                'score': str(movieObj.movie_grades),
                'date': str(movieObj.movie_year),
                'type': types,
                'img': movieObj.movie_cover,
                'url': '../detail/' + str(movieObj.movie_id)
            })
    else:
        movie_objects = Movie.objects.all().order_by('-movie_grades')[:6]  # 按豆瓣评分抽取电影
        for movieObj in movie_objects:
            type_objects = movieObj.types.all()  # return all type objects for this movie
            types = ''
            if type_objects:
                for movie_type in type_objects:
                    types = types + movie_type.type_name + ','
                types = types[:-1]

            result['data'].append({
                "title": movieObj.movie_name,
                'score': str(movieObj.movie_grades),
                'date': str(movieObj.movie_year),
                'type': types,
                'img': movieObj.movie_cover,
                'url': '../detail/' + str(movieObj.movie_id)
            })

    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')

def return_similar_movies_for_detail(request):
    current_movie_id = request.GET.get('id')
    result = {
        "data":[]
    }
    movie_id_list = RecommendSimilaryItem(int(current_movie_id))
    for movie_id in movie_id_list:
        movieObj = Movie.objects.get(movie_id=movie_id)

        type_objects = movieObj.types.all()  # return all type objects for this movie
        types = ''
        if type_objects:
            for movie_type in type_objects:
                types = types + movie_type.type_name + ','
            types = types[:-1]

        result['data'].append({
            "title": movieObj.movie_name,
            'score': str(movieObj.movie_grades),
            'date': str(movieObj.movie_year),
            'type': types,
            'img': movieObj.movie_cover,
            'url': '../detail/' + str(movieObj.movie_id)
        })
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')

def get_movies(request):
    type = request.GET.get('type')  # 获取请求的类别值
    result = {"data": []}
    print(type)
    query_set = Movie.objects.filter(types__type_name=type)[:20]
    if query_set:
        for movieObj in query_set:
            type_objects = movieObj.types.all()  # return all type objects for this movie
            types = ''
            if type_objects:
                for movie_type in type_objects:
                    types = types + movie_type.type_name + ','
                types = types[:-1]

            result['data'].append({
                "title": movieObj.movie_name,
                'score': str(movieObj.movie_grades),
                'date': str(movieObj.movie_year),
                'type': types,
                'img': movieObj.movie_cover,
                'url': '../detail/' + str(movieObj.movie_id)
            })
    # else:
    # result = {
    #     "data": [
    #         {
    #             "title": "玩具总动员",
    #             "score": "8.4",
    #             "date": "1995",
    #             "type": "喜剧,动画,家庭",
    #             # "img": "/image/MovieCover/20200131/p2557573348.webp",
    #             "img": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2220722175.jpg",
    #             "url": "../detail/1"
    #         }
    #     ]
    # }
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')

def RecommendSimilaryItem(itemID):
    '''
        根据电影ID推荐相似电影列表
    '''
    item_num = 4
    RecommendMovies0 = knn.RecommendSimilaryItems(itemID,item_num)
    RecommendMovies1 = lfm.Recommend_similary_items(itemID,item_num)
    RecommendMovies2 = mv_net.Recommend_similary_items(itemID,item_num)
    knn_weight = 0.3
    lfm_weight = 0.2
    cnn_weight = 0.5
    item_weight = {}
    for item in RecommendMovies0:
        item_weight[item] = knn_weight
    for item in RecommendMovies1:
        if item in item_weight:
            item_weight[item] += lfm_weight
        else:
            item_weight[item] = lfm_weight
    for item in RecommendMovies2:
        if item in item_weight:
            item_weight[item] += cnn_weight
        else:
            item_weight[item] = cnn_weight
    sorted(item_weight.items(),key = lambda item:item[1],reverse = True)  #按照value排序
    RecommendMoviesFinal = []
    cnt = 0
    for item in item_weight:
        RecommendMoviesFinal.append(item)
        cnt += 1
        if cnt >= 2*item_num:
            break
    # print(RecommendMoviesFinal)
    # return HttpResponse(content = RecommendMoviesFinal)
    # return JsonResponse('recommend result' : str(RecommendMoviesFinal))
    random.shuffle(RecommendMoviesFinal)
    return RecommendMoviesFinal[0:item_num]

def RecommendOtherFavoriteMovies(itemID):
    '''
        根据最喜欢这部电影的用户喜欢的电影做出推荐
    '''
    item_num = 4
    other_favorite_movies = mv_net.RecommendOtherFavoriteMovies(itemID,item_num)
    return other_favorite_movies
