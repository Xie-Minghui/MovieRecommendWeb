
import pickle
# from sklearn.externals import joblib
import joblib
# from Recommend.LFM_sql import LFM, ReadMysql
from Recommend.KNN41 import KNN, Data_process
Configuration = {
    'host': "gz-cdb-5clnvoq5.sql.tencentcdb.com",
    'port': 60720,
    'username': "root",
    'password': "MySqlxmhllly@17227016",
    'database': "db_for_project"
}
# lfm = LFM(lfm_num=10)  # lfm_num 设置模型隐向量的维度
knn = KNN()
origin_data = Data_process(
    Configuration['host'],Configuration['port'], Configuration['username'], Configuration['password'], Configuration['database'])
#   print("begin")
# try:
#     # with open('knn.pkl','rb') as f:
#         # knn = pickle.load(f.read())
#     # knn = joblib.load(r'E:\MR\UndergraduateMovieRecommand\Recommend\knn0.m')
#     with open(r'/home/UndergraduateMovieRecommand/Recommend/knn0.m', 'rb') as f: #E:\MR\UndergraduateMovieRecommand\Recommend\
#         knn = joblib.load(f)
#     # print(knn.similarity_item_matrix[:10,:60])
# except IOError:
#     print("KNN File not exist!")
# try:
#     with open(r'E:\MyProject_test\Recommend_code_origin\lfm_sql.pkl', 'rb') as f:  # E:\MyProject_test\Recommend_code_origin\
#         lfm = pickle.loads(f.read())
# except IOError:
#     print("File not exist!")
# sparse_matrix = ReadMysql(
#     Configuration['host'], Configuration['username'], Configuration['password'], Configuration['database'])

try:
    knn.denominatorA = joblib.load('/home/UndergraduateMovieRecommand/Recommend/denominatorA.m')
except IOError:
    print("File not exist!")
    
try:
    knn.denominatorB = joblib.load('/home/UndergraduateMovieRecommand/Recommend/denominatorB.m')
except IOError:
    print("File not exist!")

try:
    knn.rating_nominator = joblib.load('/home/UndergraduateMovieRecommand/Recommend/rating_nominator.m')
except IOError:
    print("File not exist!")

try:
    knn.similarity_item_matrix = joblib.load('/home/UndergraduateMovieRecommand/Recommend/similarity_item_matrix.m')
except IOError:
    print("File not exist!")
