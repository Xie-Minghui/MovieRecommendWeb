import json

from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from Users.models import User
from Movies.models import Movie
from Rating.models import Rating
from Recommend.LFM_sql import LFM, ReadMysql
from Recommend.LFM_test import lfm, sparse_matrix
from Recommend.KNN_test import knn, origin_data
from Recommend.KNN41 import KNN

from Recommend.MR.mrcnn5 import mv_network
from Recommend.MR.mrcnn5_test import mv_net 
import random

@csrf_exempt
def login(request):
    if request.method == 'POST':
        print(request.body)
        params = json.loads(request.body)
        response = HttpResponse()
        # print(params)
        if params:
            if params['params']['usr'] and params['params']['pwd']:
                usr = params['params']['usr']
                pwd = params['params']['pwd']
                print(usr + '==' + pwd)

                try:
                    usrObj = User.objects.get(user_name=usr)
                    if usrObj.password == pwd:
                        request.session['user'] = usrObj.user_id
                        response['result'] = 'true'
                        print('登陆成功')
                    else:
                        response['result'] = 'false'
                        print('密码错误')
                except:
                    response['result'] = 'false'
                    print('用户未找到')
        return response

def loginout(request):
    response = HttpResponse()
    userID = request.session.get('user', None)
    if userID:
        del request.session['user']
        response['result'] = 'true'
    
    return response

def home_page_recommend(request):
    userID = request.session.get('user', None)
    result = {
        "data":[]
    }
    if userID:
        movie_id_list = getRecommendMovies(int(userID))
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

def getRecommendMovies(userID):
    '''
        根据用户ID返回推荐电影列表
    '''
    item_num = 8
    
    RecommendMovies0, top_k_score, recommend_reasons_items = knn.ItemRecommend(
        origin_data, userID, 4, item_num)
    RecommendMovies1 = lfm.RecommendtoUser(userID, 4, sparse_matrix)
    RecommendMovies2 = mv_net.Recommend2user(userID,item_num)
    knn_weight = 0.3
    lfm_weight = 0.2
    cnn_weight = 0.5
    item_weight = {}
    for item in RecommendMovies0:
        item_weight[int(item)] = knn_weight
    for item in RecommendMovies1:
        if item in item_weight:
            item_weight[int(item)] += lfm_weight
        else:
            item_weight[int(item)] = lfm_weight
    for item in RecommendMovies2:
        if item in item_weight:
            item_weight[int(item)] += cnn_weight
        else:
            item_weight[int(item)] = cnn_weight
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
    random.shuffle(RecommendMoviesFinal)
    return RecommendMoviesFinal[0:item_num]

    # RecommendMovies = lfm.RecommendtoUser(userID, 4, sparse_matrix)
    # top_k_item, top_k_score, recommend_reasons_items = knn.ItemRecommend(
    #     origin_data, userID, 4, predict_num)
    # # # print(RecommendMovies)
    # # FinalRecommend = top_k_item + RecommendMovies
    # # print(RecommendMovies)
    # top_k_item = [int(item) for item in top_k_item]
    # FinalRecommend = top_k_item + RecommendMovies
    # # print(top_k_item)
    # # print(top_k_score)
    # # print(FinalRecommend)

    # FinalRecommend = mv_net.Recommend2user(userID,predict_num)

    # return HttpResponse(FinalRecommend)
    # return HttpResponse(RecommendMovies)
    # return HttpResponse(top_k_item)


def userInfo(request):  # 返回用户信息
    result = {
        "name": "",
        "lab": [],
        "liked": [],
        "collected": [],
        "rated": [],
    }
    userID = request.session.get('user', None)
    if userID:
        userObj = User.objects.get(user_id=userID)
        collected_movies = userObj.collected_movie.all()
        if collected_movies:  # 处理收藏的电影
            for movieObj in collected_movies:
                type_objects = movieObj.types.all()  # return all type objects for this movie
                types = ''
                if type_objects:
                    for movie_type in type_objects:
                        types = types + movie_type.type_name + ','
                    types = types[:-1]

                movie_dict = {"title": movieObj.movie_name, "score": str(movieObj.movie_grades),
                              "date": movieObj.movie_year, "type": types, "img": movieObj.movie_cover}
                result["collected"].append(movie_dict)

        rated_movies = userObj.rated_movie.all()
        if rated_movies:
            for movieObj in rated_movies:
                type_objects = movieObj.types.all()  # return all type objects for this movie
                types = ''
                if type_objects:
                    for movie_type in type_objects:
                        types = types + movie_type.type_name + ','
                    types = types[:-1]

                movie_dict = {"title": movieObj.movie_name, "score": str(movieObj.movie_grades),
                              "date": movieObj.movie_year, "type": types, "img": movieObj.movie_cover}
                result["rated"].append(movie_dict)

    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')


def ratMovie(request):
    response = HttpResponse()
    userID = request.session.get('user', None)
    if userID:
        movieID = request.GET.get('id', None)
        score = request.GET.get('score', None)
        print('评分===')
        if movieID and score:
            movieObj = Movie.objects.get(movie_id=int(movieID))
            userObj = User.objects.get(user_id=int(userID))
            try:
                Rating.objects.get(userID=userObj, movieID=movieObj)
            except:
                Rating(userID=userObj, movieID=movieObj, rating=float(score)).save()
            knn.Increment_update(int(userID),int(movieID),float(score),origin_data)
            response['result'] = 'true'
        else:
            response['result'] = 'false'
    else:
        response['result'] = 'false'
    return response


def collectMovie(request):
    response = HttpResponse()
    userID = request.session.get('user', None)
    if userID:
        movieID = request.GET.get('id', None)
        if movieID:
            movieObj = Movie.objects.get(movie_id=int(movieID))
            userObj = User.objects.get(user_id=int(userID))
            userObj.collected_movie.add(movieObj)
        else:
            response['result'] = 'false'
    else:
        response['result'] = 'false'
    return response
