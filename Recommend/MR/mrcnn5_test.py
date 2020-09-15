from Recommend.MR.mrcnn5 import DataPreprocess,mv_network
import pickle
import tensorflow as tf
import joblib

title_max_len,title_set,genres2int, features,targets,ratings,users,movies,data,movies_orig,users_orig = pickle.load(open('/home/UndergraduateMovieRecommand/Recommend/MR/dataProcess.pkl',mode='rb'))
data_features = [features, genres2int, title_set, title_max_len, movies]

mv_net=mv_network(data_features)
try:
    mv_net.model = tf.keras.models.load_model('/home/UndergraduateMovieRecommand/Recommend/MR/mv_net.h5',custom_objects = {'tf':tf})
    mv_net_info = joblib.load('/home/UndergraduateMovieRecommand/Recommend/MR/mv_net_info.m')
    mv_net.batch_size, mv_net.learning_rate, mv_net.embed_dim, mv_net.dropout_keep, mv_net.filter_num, mv_net.slide_window, mv_net.best_loss,mv_net.losses,mv_net.MODEL_DIR = \
mv_net_info[0],mv_net_info[1],mv_net_info[2],mv_net_info[3],mv_net_info[4],mv_net_info[5],mv_net_info[6],mv_net_info[7],mv_net_info[8]
    movie_matrix = pickle.load(open('/home/UndergraduateMovieRecommand/Recommend/MR/movie_matrix.p','rb'))
    user_matrix = pickle.load(open('/home/UndergraduateMovieRecommand/Recommend/MR/user_matrix.p', 'rb'))
    mv_net.movie_matrix = movie_matrix
    mv_net.user_matrix = user_matrix
    #print(mv_net.model.summary())
    # print(mv_net.Recommend_similary_items(3,5))
except IOError:
    print("Train the model for the first time!")
    mv_net.get_params(data_features)
    mv_net.training(features, targets, epochs=1)
    mv_net.slide_window = [2,3,4,5]
    mv_net_info = [mv_net.batch_size,mv_net.learning_rate,mv_net.embed_dim,mv_net.dropout_keep,mv_net.filter_num,slide_window,mv_net.best_loss,mv_net.losses,mv_net.MODEL_DIR]
    joblib.dump(mv_net_info, '/home/UndergraduateMovieRecommand/Recommend/MR/mv_net_info.m')
    #joblib.dump(mv_net,'./mv_net.m')
    mv_net.model.save('/home/UndergraduateMovieRecommand/Recommend/MR/mv_net.h5')
    # print("success")