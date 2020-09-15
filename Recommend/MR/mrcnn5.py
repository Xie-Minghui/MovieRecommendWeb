import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from tensorflow import keras
from tensorflow.python.ops import math_ops
import tensorflow as tf
import os
import pickle
import re
from urllib.request import urlretrieve
from os.path import isfile, isdir
# from tqdm import tqdm
import zipfile
import hashlib
import time
import datetime
import random
from sklearn.externals import joblib

# class DLProgress(tqdm):
#     """
#     Handle Progress Bar while Downloading
#     """
#     last_block = 0

#     def hook(self, block_num=1, block_size=1, total_size=None):
#         """
#         A hook function that will be called once on establishment of the network connection and
#         once after each block read thereafter.
#         :param block_num: A count of blocks transferred so far
#         :param block_size: Block size in bytes
#         :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
#                             a file size in response to a retrieval request.
#         """
#         self.total = total_size
#         self.update((block_num - self.last_block) * block_size)
#         self.last_block = block_num

# class ZipDownload:
#     def __init__(self):
#         None
#     def _unzip(self,save_path, _, database_name, data_path):
#         """
#         Unzip wrapper with the same interface as _ungzip
#         :param save_path: The path of the gzip files
#         :param database_name: Name of database
#         :param data_path: Path to extract to
#         :param _: HACK - Used to have to same interface as _ungzip
#         """
#         print('Extracting {}...'.format(database_name))
#         with zipfile.ZipFile(save_path) as zf:
#             zf.extractall(data_path)

#     def download_extract(self,database_name, data_path):
#         """
#         Download and extract database
#         :param database_name: Database name
#         """
#         DATASET_ML1M = 'ml-1m'

#         if database_name == DATASET_ML1M:
#             url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
#             hash_code = 'c4d9eecfca2ab87c1945afe126590906'
#             extract_path = os.path.join(data_path, 'ml-1m')  #网络中的路径
#             save_path = os.path.join(data_path, 'ml-1m.zip') #本地电脑的路径
#             extract_fn = self._unzip

#         if os.path.exists(extract_path):  #本地含有该文件，表示我之前已经下载过（程序多次运行）
#             print('Found {} Data'.format(database_name))
#             return

#         if not os.path.exists(data_path):
#             os.makedirs(data_path)

#         if not os.path.exists(save_path):
#             with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
#                 urlretrieve(
#                     url,
#                     save_path,
#                     pbar.hook)

#         assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
#             '{} file is corrupted.  Remove the file and try again.'.format(save_path)

#         os.makedirs(extract_path)
#         try:
#             extract_fn(save_path, extract_path, database_name, data_path)
#         except Exception as err:
#             shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
#             raise err

#         print('Done.')

'''
#数据预处理
1，用户数据：用户性别映射为F:0,M:1，用户年龄映射成连续的数字0-6,zip-code过滤掉
2，电影数据：movie title映射为数字，流派也映射为数字。
3，评分数据：去掉时间戳信息
'''
def DataPreprocess():
    '''
    数据预处理
    '''
    #数据的读取
    users_title = ['User-ID','Gender','Age','Occupation','Zip-code']
    users = pd.read_csv('./ml-1m/users.dat',sep = '::',header = None,names = users_title,engine = 'python')
    # users.head()

    movies_title = ['Movie-ID','Title','Genres']
    movies = pd.read_csv('./ml-1m/movies.dat',sep = '::',header = None,names = movies_title,engine = 'python')
    # movies.head()

    ratings_title = ['User-ID','Movie-ID','Rating','Timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat',sep = '::',header = None,names = ratings_title,engine = 'python')
    # ratings.head()

    #---------------------------------------------------------------------------------------------------
    #用户数据预处理
    users_title = ['UserID','Gender','Age','Occupation','Zip-code']
    users = pd.read_csv('./ml-1m/users.dat',sep = '::',header = None,names =users_title,engine = 'python')
    users = users.filter(regex = 'UserID|Gender|Age|Occupation')
    users_orig = users.values  #不知道作用是什么

    #性别数据处理
    gender_map = {'F' :0,'M':1}
    users['Gender'] = users['Gender'].map(gender_map)

    #年龄数据处理
    age_map = {val:id for id,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #电影数据预处理
    movies_title = ['MovieID','Title','Genres']
    movies = pd.read_csv('./ml-1m/movies.dat',sep = '::',header = None,names =movies_title,engine = 'python')
    movies_orig = movies.values

    #电影流派数字化
    genres_set = set()
    for genres in movies['Genres'].str.split('|'):
        genres_set.update(genres)  #使用update和add是一样的,使用add报错
    genres_set.add('<None>')
    genres2int = {val:id for id,val in enumerate(genres_set)}
    # movies['Genres'] = genres_map(genres_map)

    #将电影类型转化为等长的数字列表，长度为流派数
    genres_map = {val:[genres2int[row] for row in val.split('|')] for val in set(movies['Genres'])}
    for key in genres_map:
        for rest in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + rest,genres2int['<None>'])
    movies['Genres'] = movies['Genres'].map(genres_map)

    #电影标题除去年份
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val:pattern.match(val).group(1) for val in movies['Title']} #匹配的字符只取第一项
    movies['Title'] = movies['Title'].map(title_map)

    #电影标题数字化
    title_set = set()
    for title in movies['Title'].str.split():
        title_set.update(title)
    title_set.add('<None>')
    title2int = {val:id for id,val in enumerate(title_set)}

    title_max_len = 15
    title_map = {title:[title2int[row] for row in title.split()] for title in movies['Title']}
    for key in title_map:
        for rest in range(title_max_len - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + rest,title2int['<None>'])
    movies['Title'] = movies['Title'].map(title_map)

    #处理评分数据集
    ratings_title = ['UserID','MovieID','Rating','Timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat',sep = '::',header = None,names =ratings_title,engine = 'python')
    ratings = ratings.filter(regex = 'UserID|MovieID|Rating')

    #合并3个表
    data = pd.merge(pd.merge(ratings, users), movies)

    #分离出目标
    #将数据分成X和y两张表
    target_fields = ['Rating']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets = targets_pd.values
    pickle.dump((title_max_len,title_set,genres2int, features,targets,ratings,users,movies,data,movies_orig,users_orig),open('dataProcess.pkl','wb'))
    # return title_max_len,title_set,genres2int, features,targets,ratings,users,movies,data,movies_orig,users_orig

#构建网络模型
class mv_network(object):
    def __init__(self,data_features,batch_size = 256,learning_rate = 0.0001,embed_dim = 32,dropout_keep = 0.5,filter_num = 8,slide_window = [2,3,4,5]):
        super(mv_network,self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.dropout_keep = dropout_keep
        self.filter_num = filter_num
        self.slide_window = slide_window

        self.movie_matrix = []
        self.user_matrix = []

        self.best_loss = 99999
        self.losses = {'train':[], 'test':[]}
        self.MODEL_DIR = './models'

        uid,user_gender,user_age,user_occupation,movie_id,movie_genres,movie_titles = self.get_inputs(data_features)
        #获得用户特征
        uid_embedding_layer,user_gender_embedding_layer,user_age_embedding_layer,user_occupation_embedding_layer = \
            self.get_user_embedding_layer(uid,user_gender,user_age,user_occupation)
        user_combine_layer, user_combine_layer_flat = \
            self.get_user_features(uid_embedding_layer,user_gender_embedding_layer,user_age_embedding_layer,user_occupation_embedding_layer)

        #获得电影特征
        movieID_embedding_layer = self.get_movieID_embedding_layer(movie_id)

        movie_genres_embedding_layer = self.get_movie_generes_embedding_layer(movie_genres)

        pool_layer_flat,dropout_layer = self.get_movie_title_cnn_layer(movie_titles)

        movie_combine_layer,movie_combine_layer_flat = self.get_movie_feature_layer(movieID_embedding_layer,movie_genres_embedding_layer,dropout_layer)
        # 计算出评分
        # 将用户特征和电影特征做矩阵乘法得到一个预测评分的方案
        inference = tf.keras.layers.Lambda(lambda layer:tf.reduce_sum(layer[0] * layer[1],axis = 1),name='inference')((user_combine_layer_flat,movie_combine_layer_flat))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)
        
        self.model = tf.keras.Model(
            inputs = [uid,user_gender,user_age,user_occupation,movie_id,movie_genres,movie_titles],
            outputs = [inference])
        
        self.model.summary()    #打印参数信息

        #定义评价指标和优化器
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        #定义损失函数
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetricsMAE = tf.keras.metrics.MeanAbsoluteError()
        self.avg_loss = tf.keras.metrics.Mean('loss',dtype = tf.float32)

        #保存参数
        if tf.io.gfile.exists(self.MODEL_DIR ):
            pass
        else:
            tf.io.gfile.mkdir(self.MODEL_DIR )
        checkpoint_dir = os.path.join(self.MODEL_DIR , 'checkpoints') 
        self.checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
        self.checkpoint = tf.train.Checkpoint(model = self.model,optimizer = self.optimizer)
        #重构代码，如果存在一个checkpoint
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    def get_params(self,data_features):
        features, genres2int, title_set, title_max_len, movies = data_features[0],data_features[1],data_features[2],data_features[3],data_features[4]
        reserved_num = 1
        #用户id个数
        self.uid_max = max(features.take(0,1)) + reserved_num
        #性别个数
        self.gender_max = 2
        #年龄分类个数
        self.age_max = max(features.take(3,1)) + 1
        #职业个数
        self.occupation_max = max(features.take(4,1)) + 1

        #电影
        #电影id个数
        self.movieid_max = max(features.take(1,1)) + reserved_num
        #电影流派
        self.genres_max = max(genres2int.values()) + 1
        #标题单词个数
        self.title_max = len(title_set)
        #标题文本最长长度
        self.sentences_size = title_max_len
        #电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
        self.movieid2idx = {val[0]:id for id, val in enumerate(movies.values)}


    #定义输入的占位符
    def get_inputs(self,data_features):
        self.get_params(data_features)
        uid = tf.keras.layers.Input(shape=(1,),dtype="int32",name = 'uid')
        user_gender = tf.keras.layers.Input(shape = (1,),dtype = "int32",name = 'user_gender')
        user_age = tf.keras.layers.Input(shape = (1,),dtype = "int32",name = 'user_age')
        user_occupation = tf.keras.layers.Input(shape = (1,),dtype = "int32",name = 'user_occupation')

        movie_id = tf.keras.layers.Input(shape = (1,),dtype = "int32",name = 'movie_id')
        movie_genres = tf.keras.layers.Input(shape = (self.genres_max-1,),dtype = "int32",name = 'movie_genres')
        movie_titles = tf.keras.layers.Input(shape = (self.sentences_size,),dtype = "string",name = 'movie_titles')

        return uid,user_gender,user_age,user_occupation,movie_id,movie_genres,movie_titles

    #定义用户的嵌入矩阵
    def get_user_embedding_layer(self,uid,user_gender,user_age,user_occupation):
        uid_embedding_layer = tf.keras.layers.Embedding(self.uid_max,self.embed_dim,input_length = 1,name = 'uid_embedding_layer')(uid)
        user_gender_embedding_layer = tf.keras.layers.Embedding(self.gender_max,self.embed_dim//8,input_length = 1,name ='user_gender_embedding_layer')(user_gender)
        user_age_embedding_layer = tf.keras.layers.Embedding(self.age_max,self.embed_dim//2,input_length = 1,name ='user_age_embedding_layer')(user_age)
        user_occupation_embedding_layer = tf.keras.layers.Embedding(self.occupation_max,self.embed_dim,input_length = 1,name ='user_occupation_embedding_layer')(user_occupation)

        # movie_id_embedding_layer = tf.keras.layers.Embedding(movieid_max,self.embed_dim,input_length = 1,name ='movie_id_embedding_layer')
        # movie_genres_embedding_layer = tf.keras.layers.Embedding(genres_max,self.embed_dim,input_length = 18,name ='movie_genres_embedding')
        return uid_embedding_layer,user_gender_embedding_layer,user_age_embedding_layer,user_occupation_embedding_layer

    #将用户的嵌入矩阵先各个经过一个全连接层，然后整合到一起，然后再连接一个全连接层
    def get_user_features(self,uid_embedding_layer,user_gender_embedding_layer,user_age_embedding_layer,user_occupation_embedding_layer):
        #各自全连接
        uid_fc_layer = tf.keras.layers.Dense(self.embed_dim,name = 'uid_fc_layer',activation ='relu')(uid_embedding_layer)
        user_gender_fc_layer = tf.keras.layers.Dense(self.embed_dim,name = 'user_gender_fc_layer',activation ='relu')(user_gender_embedding_layer)
        user_age_fc_layer = tf.keras.layers.Dense(self.embed_dim,name ='user_age_fc_layer',activation ='relu')(user_age_embedding_layer)
        user_occupation_fc_layer = tf.keras.layers.Dense(self.embed_dim,name = 'user_occupation_fc_layer',activation ='relu')(user_occupation_embedding_layer)
        #将用户特征合并
        user_combine_layer0 = tf.keras.layers.concatenate([uid_fc_layer,user_gender_fc_layer,user_age_fc_layer,user_occupation_fc_layer])
        user_combine_layer1 = tf.keras.layers.Dense(256,activation='tanh')(user_combine_layer0)
        #将得到的用户特征展平
        user_combine_layer_flat = tf.keras.layers.Reshape([256],name='user_combine_layer_flat')(user_combine_layer1)

        return user_combine_layer1, user_combine_layer_flat

    #定义movieID的嵌入矩阵    
    def get_movieID_embedding_layer(self,movie_id):
        movieID_embedding_layer = tf.keras.layers.Embedding(self.movieid_max,self.embed_dim,input_length = 1,name = 'movieID_embedding_layer')(movie_id)
        return movieID_embedding_layer

    def get_movie_generes_embedding_layer(self,movie_genres):
        movie_genres_embedding_layer = tf.keras.layers.Embedding(self.genres_max,self.embed_dim,input_length = self.genres_max,name = 'movie_genres_embedding_layer')(movie_genres)
        movie_genres_embedding_layer = tf.keras.layers.Lambda(lambda layer : tf.reduce_sum(layer,axis=1,keepdims=True))(movie_genres_embedding_layer)
        #不知道为什么要将电影流派求和，每个电影的流派特征变成一个数
        return movie_genres_embedding_layer

    #movie title的文本卷积网络的实现
    def get_movie_title_cnn_layer(self,movie_titles):
        #从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
        movie_title_embedding_layer = tf.keras.layers.Embedding(self.title_max,self.embed_dim,input_length = self.sentences_size,name = 'movie_titles_embedding_layer')(movie_titles)
        sp = movie_title_embedding_layer.shape
        movie_title_embedding_layer_expand = tf.keras.layers.Reshape([sp[1],sp[2],1])(movie_title_embedding_layer)
        #对文本嵌入层使用不同尺寸的卷积核进行卷积核和最大池化
        pool_layer_list = []
        for slide_window_size in self.slide_window:
            conv_layer = tf.keras.layers.Conv2D(self.filter_num,(slide_window_size,self.embed_dim),1,activation = 'relu')(movie_title_embedding_layer_expand)
            maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size = (self.sentences_size-slide_window_size+1,1),strides=1)(conv_layer)
            pool_layer_list.append(maxpool_layer)
        
        #Dropout层
        pool_layer = tf.keras.layers.concatenate(pool_layer_list,3,name = 'pool_layer')
        max_num = len(self.slide_window) * self.filter_num
        pool_layer_flat = tf.keras.layers.Reshape([1,max_num],name = 'pool_layer_flat')(pool_layer)
        dropout_layer = tf.keras.layers.Dropout(self.dropout_keep,name = 'dropout_layer')(pool_layer_flat)

        return pool_layer_flat,dropout_layer

    def get_movie_feature_layer(self,movieID_embedding_layer,movie_genres_embedding_layer,dropout_layer):
        #各嵌入层的全连接
        movieID_fc_layer = tf.keras.layers.Dense(self.embed_dim,name = 'movieID_fc_layer',activation ='relu')(movieID_embedding_layer)
        movie_genres_fc_layer = tf.keras.layers.Dense(self.embed_dim,name = 'movie_genres_fc_layer',activation ='relu')(movie_genres_embedding_layer)
        #将嵌入层全连接层输出和文本卷积网络合并，然后再接一个全连接层
        movie_combine_layer = tf.keras.layers.concatenate([movieID_fc_layer,movie_genres_fc_layer,dropout_layer],2)
        movie_combine_layer = tf.keras.layers.Dense(256,name = 'movie_combine_layer',activation ='tanh')(movie_combine_layer)
        movie_combine_layer_flat = tf.keras.layers.Reshape([256],name='movie_combine_layer_flat')(movie_combine_layer)

        return movie_combine_layer,movie_combine_layer_flat
    #取得batch
    def get_batches(self,Xs, ys):
        for start in range(0, len(Xs), self.batch_size):
            end = min(start + self.batch_size, len(Xs))
            yield Xs[start:end], ys[start:end]
    #计算图构建，结果计算
    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6]], training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(labels, logits)
            
            # metrics = self.compute_metrics(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.ComputeMetricsMAE(y, logits)#
        
        self.avg_loss(loss)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

    #训练集的训练
    def training(self,users,movies,features,target_values,epochs = 1,log_freq = 50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        for epoch_i in range(epochs):
            train_X,test_X,train_y,test_y = train_test_split(features,target_values,test_size = 0.2,random_state = 0)
            train_batches = self.get_batches(train_X,train_y)
            batch_num = len(train_X) // self.batch_size

            train_start = time.time()

            if True:
                start = time.time()
                

                for batch_i in range(batch_num):
                    x,y = next(train_batches)
                    #标题和电影类别特殊处理
                    movie_genres = np.zeros([self.batch_size,18])
                    movie_titles = np.zeros([self.batch_size,15])
                    for i in range(self.batch_size):
                        movie_genres[i] = x.take(6,1)[i]
                        movie_titles[i] = x.take(5,1)[i]

                    loss,logits = self.train_step([np.reshape(x.take(0,1),[self.batch_size,1]).astype(np.float32),
                                                   np.reshape(x.take(2,1),[self.batch_size,1]).astype(np.float32),
                                                   np.reshape(x.take(3,1),[self.batch_size,1]).astype(np.float32),
                                                   np.reshape(x.take(4,1),[self.batch_size,1]).astype(np.float32),
                                                   np.reshape(x.take(1,1),[self.batch_size,1]).astype(np.float32),
                                                   movie_genres.astype(np.float32),
                                                   movie_titles.astype(np.float32)],
                                                   np.reshape(y,[self.batch_size, 1]).astype(np.float32))                
                    
                    # avg_loss(loss)  #计算平均误差

                    self.losses['train'].append(loss)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('avg_loss',self.avg_loss.result(),step = epoch_i)
                        tf.summary.scalar('MAE',self.ComputeMetricsMAE.result(),step = epoch_i)
                    
                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        #                         summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)
                        #                         summary_ops_v2.scalar('mae', self.ComputeMetricsMAE.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', avg_mae.result(), step=self.optimizer.iterations)

                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetricsMAE.result()), rate))
                        # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        #     self.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
#                         self.avg_loss.reset_states()
#                         self.ComputeMetricsMAE.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()
                self.avg_loss.reset_states()
                self.ComputeMetricsMAE.reset_states()
            train_end = time.time()
            print(
                '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                         train_end - train_start))
            #             with self.test_summary_writer.as_default():
            self.testing((test_X, test_y), self.optimizer.iterations)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(self.MODEL_DIR , 'export')
        tf.saved_model.save(self.model, self.export_path)  #保存模型的参数
        self.GenerateMovieModel(movies)
        self.GenerateUserModel(users)
        
    def testing(self, test_dataset, step_num):
        test_X, test_y = test_dataset
        test_batches = self.get_batches(test_X, test_y)

        """Perform an evaluation of `model` on the examples from `dataset`."""
#         avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        #         avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, 18])
            for i in range(self.batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([self.batch_size, self.sentences_size])
            for i in range(self.batch_size):
                titles[i] = x.take(5, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                 categories.astype(np.float32),
                                 titles.astype(np.float32)], training=False)
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            self.avg_loss(test_loss)
            # 保存测试损失
            self.losses['test'].append(test_loss)
            self.ComputeMetricsMAE(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            # avg_loss(self.compute_loss(labels, logits))
            # avg_mae(self.compute_metrics(labels, logits))

        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(self.avg_loss.result(), self.ComputeMetricsMAE.result()))
        # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
        #         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        #         summary_ops_v2.scalar('mae', self.ComputeMetricsMAE.result(), step=step_num)
        # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)

        if self.avg_loss.result() < self.best_loss:
            self.best_loss = self.avg_loss.result()
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)
    
    def rating_movie(self,userID,movieID,users):
        
        genres = np.array(movies.values[self.movieid2idx[movieID]][2]).reshape(1,self.genres_max)
        titles = np.array(movies.values[self.movieid2idx[movieID]][1]).reshape(1,self.sentences_size)

        inference_val = self.model([np.reshape(users.values[userID-1][0],[1,1]),
                                    np.reshape(users.values[userID-1][1],[1,1]),
                                    np.reshape(users.values[userID-1][2],[1,1]),
                                    np.reshape(users.values[userID-1][3],[1,1]),
                                    np.reshape(movies.values[self.movieid2idx(movieID)][0],[1,1]),
                                    genres,
                                    titles])
        return inference_val.numpy()

    def GenerateMovieModel(self,movies):
        print("MovieModel")
        movie_layer_model = tf.keras.models.Model(inputs = [self.model.input[4],self.model.input[5],self.model.input[6]],
                                                outputs = self.model.get_layer("movie_combine_layer_flat").output)           
        self.movie_matrix = []   #                                      
        for item in movies.values:
            genre = np.reshape(item.take(2),[1,self.genres_max-1])
            title = np.reshape(item.take(1),[1,self.sentences_size]) 
            movie_combine_layer_flat_val = movie_layer_model([np.reshape(item.take(0),[1,1]),genre,title])
            self.movie_matrix.append(movie_combine_layer_flat_val)
        pickle.dump(np.array(self.movie_matrix).reshape(-1,256),open('movie_matrix.p','wb'))

    def GenerateUserModel(self,users):
        print("UserModel")
        user_layer_model = tf.keras.models.Model(inputs = [self.model.input[0],self.model.input[1],self.model.input[2],self.model.input[3]],
                                                outputs = self.model.get_layer("user_combine_layer_flat").output)
        self.user_matrix = []        #用户的特征矩阵
        for item in users.values:
            user_combine_layer_flat_val = user_layer_model([np.reshape(item.take(0),[1,1]),np.reshape(item.take(1),[1,1]),np.reshape(item.take(2),[1,1]),np.reshape(item.take(3),[1,1])])
            self.user_matrix.append(user_combine_layer_flat_val)
        pickle.dump(self.user_matrix,open('user_matrix.p','wb'))
        #user_matrix = pickle.load(open('user_matrix.p',mod = 'rb'))

    def Recommend_similary_items(self,movieID,topK):
        """
            下面的算法类似LFM的算法，推荐相似的物品
        """
        normed_movie_matrix = tf.sqrt(tf.reduce_sum(tf.square(self.movie_matrix),1,keepdims = True)) 
        normalized_movie_matrix = self.movie_matrix / normed_movie_matrix
        
        #print(self.movieid2idx)
        movie_embed_vector = self.movie_matrix[self.movieid2idx[movieID]].reshape(1,256)
        movies_similarity = tf.matmul(movie_embed_vector,tf.transpose(normalized_movie_matrix))
        movies_similarity_arr = movies_similarity.numpy()

        p = np.squeeze(movies_similarity_arr) #删除维度为0的维度

        random_times = 4
        movies_similary_loc = np.argpartition(-p,topK*random_times)
        random.shuffle(movies_similary_loc[0:topK*random_times])
        recommend_movies = movies_similary_loc[0:topK]  #存在的问题：不清楚movie_matrix特征矩阵这里面的位置是否对应真正的电影标号
        recommend_movies = [x+1 for x in recommend_movies]
        return recommend_movies
    
    def Recommend2user(self,userID,ItemNum):
        '''
            推荐你喜欢的电影
        '''
        # user_embed = self.user_matrix[userID-1].reshape([1,256])
        
        user_ratings = tf.matmul(self.user_matrix[userID-1], tf.transpose(self.movie_matrix))
        user_ratings = np.array(user_ratings)[0]
        # print(user_ratings)
        random_times = 4
        movies_favorite_loc = np.argpartition(-user_ratings,ItemNum*random_times)[0:ItemNum*random_times]
        random.shuffle(movies_favorite_loc)
        movies_favorite = movies_favorite_loc[0:ItemNum]
        movies_favorite = [x+1 for x in movies_favorite]
        return movies_favorite
    
    def RecommendOtherFavoriteMovies(self,movieID,itemNum):
        '''
            坑点：直接进行类型转换会导致程序崩溃
        '''

        movie_embed_vector = self.movie_matrix[self.movieid2idx[movieID]].reshape((1,256))

        movie_mat = np.zeros((len(self.movie_matrix),1,self.movie_matrix[0].shape[0]))
        for i in range(len(self.movie_matrix)):
            movie_mat[i] = self.movie_matrix[i]
        movie_mat = np.squeeze(movie_mat)


        user_ratings = np.matmul(movie_embed_vector , movie_mat.T)
        user_ratings = np.array(user_ratings)
        random_times = 4

        user_favorite_loc = np.argpartition(-user_ratings,itemNum*random_times)[0:itemNum*random_times]
        random.shuffle(user_favorite_loc)
        user_favorite = user_favorite_loc[0,0:itemNum]

        user_mat2 = list(movie_mat[list(user_favorite)])
        other_favorite = np.matmul(user_mat2,movie_mat.T)

        other_favorite_loc = np.argpartition(-other_favorite,itemNum)[0,0:itemNum]
        other_favorite_movies = [x+1 for x in other_favorite_loc]

        return other_favorite_movies

# #结果的可视化
def test():
    title_max_len,title_set,genres2int, features,targets,ratings,users,movies,data,movies_orig,users_orig = pickle.load(open('dataProcess.pkl',mode='rb'))
    data_features = [features, genres2int, title_set, title_max_len, movies]
    mv_net=mv_network(data_features)
    try:
        mv_net.model = tf.keras.models.load_model('mv_net.h5')
        mv_net_info = joblib.load('./mv_net_info.m')
        mv_net.batch_size,mv_net.learning_rate,mv_net.embed_dim,mv_net.dropout_keep,mv_net.filter_num,mv_net.slide_window,mv_net.best_loss,mv_net.losses,mv_net.MODEL_DIR = \
    mv_net_info[0],mv_net_info[1],mv_net_info[2],mv_net_info[3],mv_net_info[4],mv_net_info[5],mv_net_info[6],mv_net_info[7],mv_net_info[8]
        movie_matrix = pickle.load(open('movie_matrix.p','rb'))
        user_matrix = pickle.load(open('user_matrix.p', 'rb'))
        mv_net.movie_matrix = movie_matrix
        mv_net.user_matrix = user_matrix
        #print(mv_net.model.summary())
        print(mv_net.Recommend_similary_items(3,5))
    except IOError:
        print("Train the model for the first time!")
        mv_net.get_params(data_features)
        mv_net.training(users,movies,features, targets, epochs=1,log_freq = 50)
        print("train end")
        slide_window = [2,3,4,5]
        mv_net_info = [mv_net.batch_size,mv_net.learning_rate,mv_net.embed_dim,mv_net.dropout_keep,mv_net.filter_num,mv_net.slide_window,mv_net.best_loss,mv_net.losses,mv_net.MODEL_DIR]
        joblib.dump(mv_net_info, './mv_net_info.m')
        #joblib.dump(mv_net,'./mv_net.m')
        mv_net.model.save('mv_net.h5')
        print("success")

if __name__ == '__main__':
    test()
