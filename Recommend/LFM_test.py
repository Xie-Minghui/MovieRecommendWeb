
import pickle
from Recommend.LFM_sql import LFM, ReadMysql
# from Recommend_code_origin.LFM_sql import LFM
# Configuration = {
#     'host': "localhost",
#     'username': "root",
#     'password': "112803",
#     'database': "mrtest"
# }
Configuration = {
    'host': "gz-cdb-5clnvoq5.sql.tencentcdb.com",
    'port': 60720,
    'username': "root",
    'password': "MySqlxmhllly@17227016",
    'database': "db_for_project"
}

sparse_matrix = ReadMysql(
    Configuration['host'], Configuration['port'], Configuration['username'], Configuration['password'], Configuration['database'])
try:
    with open(r'/home/UndergraduateMovieRecommand/Recommend/lfm_info.pkl', 'rb') as f: #E:\MR\UndergraduateMovieRecommand\Recommend\
        lfm_info = pickle.loads(f.read())
except IOError:
    print("File not exist!")
try:
    with open(r'/home/UndergraduateMovieRecommand/Recommend/lfm_Up.pkl', 'rb') as f: #E:\MR\UndergraduateMovieRecommand\Recommend\
        lfm_Up = pickle.loads(f.read())
except IOError:
    print("File not exist!")

try:
    with open(r'/home/UndergraduateMovieRecommand/Recommend/lfm_VTp.pkl', 'rb') as f: #E:\MR\UndergraduateMovieRecommand\Recommend\
        lfm_VTp = pickle.loads(f.read())
except IOError:
    print("File not exist!")

lfm = LFM(lfm_num=10)  # lfm_num 设置模型隐向量的维度
# try:
#     with open(r'/home/UndergraduateMovieRecommand/Recommend/lfm_sql.pkl', 'rb') as f: #E:\MR\UndergraduateMovieRecommand\Recommend\
#         lfm = pickle.loads(f.read())
# except IOError:
#     print("File not exist!")
lfm.Up = lfm_Up
lfm.VTp = lfm_VTp
[lfm.lfm_num, lfm.alpha, lfm.lambda_u,
          lfm.lambda_i, lfm.r_max, lfm.c_max] = lfm_info
