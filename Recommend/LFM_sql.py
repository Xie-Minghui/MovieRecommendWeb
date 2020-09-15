from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import numpy as np
import math
import random
import copy
import pickle
import os
import pymysql
import operator

class LFM:
    def __init__(self,lfm_num = 0):
        self.lfm_num = lfm_num  #隐向量的个数
        self.alpha = None   #学习率
        self.lambda_u = None #用户的正则化稀疏
        self.lambda_i = None #物品的正则化稀疏
        self.Up = None  #用户偏差矩阵
        self.VTp = None #物品偏差矩阵
        self.r_max = None #行号最大值
        self.c_max = None #列号最大值

    def Sparse_matrix2rcd(self,sparse_matrix)->np.array:
        """
            输入：稀疏矩阵
            输出：rcd存储形式（行号，列号，值）
            功能：将稀疏矩阵转化为普通形式，方便分离训练集与测试集，因为目前还没有找到直接对稀疏矩阵分离的方法
        """
        indptr = sparse_matrix.indptr
        indices = sparse_matrix.indices
        data = sparse_matrix.data
        row,col,mydata = [],[],[]
        len_ind = len(indptr)

        indptr = indptr.tolist()
        indices = indices.tolist()
        data = data.tolist()

        for r in range(len_ind-1):
            col_list = indices[indptr[r]:indptr[r+1]]
            len_col_list = len(col_list)
            if len_col_list == 0:
                continue
            col = col + col_list
            row = row + [r for i in range(len_col_list)]  #从0开始
            mydata = mydata + data[indptr[r]:indptr[r+1]]
        return np.array(row),np.array(col),np.array(mydata)


    def Cal_deviation(self,sparse_matrix_train):
        """
            输入：训练的稀疏矩阵
            输出：用户偏差和物品偏差的初始值
            功能：利用稀疏矩阵求用户偏差和物品偏差，使用均值作为用户偏差和物品偏差的初始值
        """
        sum_row = sparse_matrix_train.sum(axis=1) #求每行的和，返回np矩阵 2000*1
        sum_col = sparse_matrix_train.sum(axis=0) #求每列的和，返回np矩阵 1*2000

        r_nonzero,c_nonzero = sparse_matrix_train.nonzero()
        len_nonzero = len(r_nonzero)
        r_max = np.max(r_nonzero)  #最大用户编号
        c_max = np.max(c_nonzero)  #最大物品编号
        user_deviation,item_deviation = np.zeros(r_max+1),np.zeros(c_max+1)
        item_nonzero = np.zeros(len_nonzero)
        cnt = 1
        for i in range(1,len_nonzero):
            if(r_nonzero[i] != r_nonzero[i-1]):
                user_deviation[r_nonzero[i-1]] = sum_row[r_nonzero[i-1],0] / cnt
                cnt = 1
            else:
                cnt += 1
        
        for i in range(0,len_nonzero):
            item_nonzero[c_nonzero[i]] += 1
        for i in range(0,c_max+1):
            if(item_nonzero[i] != 0):
                item_deviation[i] = sum_col[0,i] / item_nonzero[i]
        
        return user_deviation,item_deviation

    def Mean_centered(self,row_train,col_train,data_train,user_deviation,item_deviation,r_max,c_max):
        """
            输入：行号，列号，值，用户偏差，物品偏差，行号最大值，列号最大值
            输出：均值中心化之后的稀疏矩阵
            功能：均值中心化，每个用户减去自己评分的均值，每个物品减去自己评分的均值
        """
        len_data = len(data_train)
        data_train2 = copy.deepcopy(data_train)
        for i in range(len_data):
            data_train2[i] -= (user_deviation[row_train[i]] + item_deviation[col_train[i]])
        return csr_matrix((data_train2,(row_train,col_train)),shape = (r_max+1,c_max+1))

    def getUp(self,U,user_deviation):
        """
            输入：用户降维后的矩阵，用户偏差
            返回：附加用户偏差后的用户矩阵
            功能：在潜在因子模型嵌入用户偏差
        """
        user_deviation = user_deviation.reshape(len(user_deviation),1)  #不会原地改变
        Up = np.append(U,user_deviation,axis = 1)
        m,k = Up.shape
        Ones = np.ones((m,1))
        Up = np.append(Up,Ones,axis = 1)
        return Up

    def getVTp(self,VT,item_deviation):
        """
            输入：物品降维后的矩阵，物品偏差
            返回：附加物品偏差后的物品矩阵
            功能：在潜在因子模型嵌入物品偏差
        """
        item_deviation = item_deviation.reshape(1,len(item_deviation))
        k,n = VT.shape
        Ones = np.ones((1,n))
        VTp = np.append(VT,Ones,axis = 0)
        VTp = np.append(VTp,item_deviation,axis = 0)
        return VTp

    #随机梯度下降
    def Gradient_descent(self,rcd_train,train_times):
        """
            输入：(行号，列号，值),训练次数
            返回：无
            功能：随机梯度下降法学习用户和物品的隐向量
        """
        m,k = self.Up.shape
        k,n = self.VTp.shape
        Ui,VTi = np.zeros((m,k)),np.zeros((k,n))
        c = rcd_train[0].shape[0]
        print("训练矩阵的大小：",m,n)
        shuffle_list = [i for i in range(c)]  
        # print("初始值")
        # print(self.Up)
        # print(self.VTp)
        for l in range(train_times):
            print(l)
            random.shuffle(shuffle_list) #使得更新顺序随机化
            # rcd_train = rcd_train[:,shuffle_list]
            rcd_train_shuffle = copy.deepcopy(rcd_train)
            for i in range(c):
                rcd_train_shuffle[:,i] = rcd_train[:,shuffle_list[i]]
            rcd_train = copy.deepcopy(rcd_train_shuffle)

            for i in range(c-1):  #根据打乱的训练数据学习
                # print(type(self.Up[int(rcd_train[0,i]),:]))
                # print(self.Up[int(rcd_train[0,i]),:].shape)
                predict_rating = sum(self.Up[int(rcd_train[0,i]),:] * self.VTp[:,int(rcd_train[1,i])])
                predict_rating1 = max(min(predict_rating,5.0),0.0)
                e = (rcd_train[2][i] - predict_rating1)  #这里的乘法为向量内积
                
                # print('e: ',e,rcd_train[2][i],predict_rating1)
                for j in range(k):
                    Ui[int(rcd_train[0,i]),j] = self.Up[int(rcd_train[0,i]),j] + self.alpha * (e*self.VTp[j,int(rcd_train[1,i])] - self.lambda_u*self.Up[int(rcd_train[0,i]),j])
                    VTi[j,int(rcd_train[1,i])] = self.VTp[j,int(rcd_train[1,i])] + self.alpha * (e*self.Up[int(rcd_train[0,i]),j]  - self.lambda_i*self.VTp[j,int(rcd_train[1,i])])
                for j in range(k):
                    self.Up[int(rcd_train[0,i]),j] = Ui[int(rcd_train[0,i]),j]
                    self.VTp[j,int(rcd_train[1,i])] = VTi[j,int(rcd_train[1,i])]
            self.Up[:,k-1] = np.ones(m)
            self.VTp[k-2,:] = np.ones(n)

        return None
    
    def Fit(self,sparse_matrix):
        """
            参数：稀疏矩阵的路径
            返回值：训练集和测试集
            功能：将稀疏矩阵的路径喂给模型
        """
        # sparse_matrix = load_npz(npz_path)  #需要绝对路径
        row,col,data = self.Sparse_matrix2rcd(sparse_matrix)
        len_row = len(row)
        row = row.reshape(len_row,1)
        col = col.reshape(len_row,1)
        self.r_max = np.max(row)
        self.c_max = np.max(col)

        # self.lfm_num = int(math.pow(self.c_max,0.33))  #设置隐变量的个数
        X = np.append(row,col,axis = 1)
        X_train,X_test,y_train,y_test = train_test_split(X,data,test_size = 0.3,random_state = 42)
        return X_train,X_test,y_train,y_test
        
    def Train(self,X_train,y_train, alpha = 0.0001,lambda_u = 0.25,lambda_i = 0.25,train_times = 100):
        self.alpha, self.lambda_u,self.lambda_i = alpha,lambda_u, lambda_i

        sparse_matrix_train = csr_matrix((y_train,(X_train[:,0],X_train[:,1])),shape = (self.r_max+1,self.c_max+1))
        user_deviation,item_deviation = self.Cal_deviation(sparse_matrix_train) #求用户偏差和物品偏差

        row_train,col_train,data_train = self.Sparse_matrix2rcd(sparse_matrix_train)
        #均值中心化
        sparse_matrix_train_mean = self.Mean_centered(row_train,col_train,data_train,user_deviation,item_deviation,self.r_max,self.c_max)

        #第一次训练使用SVD分解初始化用户和物品的隐向量
        if self.Up is None:
            U0,Sigma,VT = randomized_svd(sparse_matrix_train_mean,n_components = self.lfm_num)
            U = U0*Sigma
            #矩阵拓展引入用户偏差和物品偏差
            self.Up = self.getUp(U,user_deviation)
            self.VTp = self.getVTp(VT,item_deviation)
            print("第一次训练！")
        
        print(self.Up.shape,self.VTp.shape)  #(944, 101) (101, 1683)
        # print(self.Up)
        rcd_train = np.array([row_train,col_train,data_train])  #rcd训练数据，未均值中心化,列表里面是数组

        #计算训练之前的评分总误差
        S = np.dot(self.Up,self.VTp)
        sum0 = 0.0
        for i in range(len(rcd_train)):
            # if  i < 10:
            #     print(rcd_train[2,i], S[int(rcd_train[0,i]),int(rcd_train[1,i])])
            sum0 += math.fabs(rcd_train[2,i] - S[int(rcd_train[0,i]),int(rcd_train[1,i])])
        print("训练前的在训练集上的总评分误差： {0}".format(sum0))  #3.06
        #注意控制评分的边界

        #开始训练
        self.Gradient_descent(rcd_train,train_times)

    
    def Get_RMSEandMAE(self,Rate,X_test,y_test):
        """
            参数：：评分矩阵，测试集
            得到在测试集上的RMSE
        """
        r,c = X_test.shape
        RMSE = 0.0
        MAE = 0.0
        for i in range(r):
            if Rate[X_test[i,0],X_test[i,1]] > 5:
                Rate[X_test[i,0],X_test[i,1]] = 5.0
            if Rate[X_test[i,0],X_test[i,1]] <= 0:
                Rate[X_test[i,0],X_test[i,1]] = 0.0
            RMSE += (y_test[i] - Rate[X_test[i,0],X_test[i,1]])**2
            MAE += math.sqrt(y_test[i] - Rate[X_test[i,0],X_test[i,1]])
        return math.sqrt(RMSE/r),MAE/r
    

    def RecommendtoUser(self,user,item_num,sparse_matrix):
        '''
            输入：用户ID，推荐物品数量，稀疏矩阵
            输出：用户推荐物品列表
            功能：给特定用户推荐物品，添加了随机因素，保证每次推荐的电影不完全相同,
            选出前item_num*random_times个物品，然后随机从这个列表选出item_num个。
        '''
        #得到用户对各个物品的预测评分

        user_rating = np.dot(self.Up[user],self.VTp) #(1,13) (13,1683)
        user_rating_len = len(user_rating)
        userAll = sparse_matrix.getrow(user)
        userAll = userAll.toarray()[0,0:user_rating_len]
        userOneHot = np.array(list(map(lambda x:(0 if x > 0 else 1),userAll)))
        user_rating_rest = np.multiply(user_rating,userOneHot) #得到用户没有评分的,对应位置相乘 (1683,) (1683,)

        #得到topK
        random_times = 4
        #--------------------------------------------
        # user_rating_rest_loc = np.array([-user_rating_rest,np.array([x for x in range(1,user_rating_len+1)])])        
        # user_rating_rest_loc_sorted = np.argpartition(user_rating_rest_loc,item_num*random_times,axis = 1)
        # recommend_items = user_rating_rest_loc_sorted[0,0:item_num*random_times]
        # random.shuffle(recommend_items)
        #--------------------------------------
        recommend_items_loc = np.argpartition(-user_rating_rest,item_num*random_times)
        # print(recommend_items_loc.shape)
        #recommend_items_loc经过multiply得到的还是向量,经过函数argpartition的位置还是向量，不同于下一个函数cosineSimilarity_loc的ndarray
        random.shuffle(recommend_items_loc[0:item_num*random_times])
        recommend_items = recommend_items_loc[0:item_num]
        recommend_items = np.array([x+1 for x in recommend_items])
        return recommend_items[0:item_num]

    def Recommend_similary_items(self,itemID,item_num):
        """
            输入：物品ID，推荐的数量
            输出: 推荐与当前物品相似的物品列表
            功能：推荐与当前物品相似的物品，添加了随机因素，使用了物品之间的余弦相似度
        """
        #使用矩阵乘法算余弦相似度
        item_vector = self.VTp[:,itemID]
        item_vector = item_vector.reshape(1,len(item_vector))
        VTp0 = np.sqrt(np.sum(pow(self.VTp.T,2),axis = 1))  #(1683,)
        r0 = len(VTp0)
        allItem_vector = self.VTp.T/VTp0.reshape(r0,1)  #(1683,13)
        cosineSimilarity = np.matrix(item_vector) * np.matrix(allItem_vector.T) #(1,13) (13,1683),需将ndarray转化为矩阵，然后矩阵乘法
        #排序
        cosineSimilarity = np.array(cosineSimilarity)
        random_times = 4  #随机因子 
        cosineSimilarity_loc = np.argpartition(-cosineSimilarity,item_num*random_times)  #(1,1683)
        #shuffle函数对array有用，对列表似乎无效
        random.shuffle(cosineSimilarity_loc[0,0:item_num*random_times]) #原地改变，返回NoneType
        similarity_items = cosineSimilarity_loc[0,0:item_num]
        similarity_items = np.array([x+1 for x in similarity_items])
        return similarity_items

    def getCoverage(self,X_test,sparse_matrix,predict_num):
        '''
            返回覆盖率和基尼指数
        '''
        recommend_list = []
        for item in X_test:
            topkItems = self.RecommendtoUser(item[0],predict_num,sparse_matrix)
            recommend_list = recommend_list + topkItems
        
        coverage = len(set(recommend_list)) / self.itemMax  #推荐物品的覆盖率
        times = {}
        for i in set(recommend_list):
            times[i] = recommend_list.count(i)
        j = 1
        n = len(times)
        G = 0
        for item,weight in sorted(times.items(),key=operator.itemgetter(1)):
            G += (2*j-n-1)*weight
        return coverage, G/float(n-1)

    #训练之后的测试集的总误差
    def lfm_test(self, X_test, y_test):
        # print("最后的训练结果：")
        # print(self.Up)
        # print(self.VTp)
        Rate = np.dot(self.Up, self.VTp)
        sum1 = 0.0
        for i in range(len(y_test)):
            # if i < 10:
            #     print( Rate[int(X_test[i,0]),int(X_test[i,1])]) #-------------------------------
            sum1 += math.fabs(y_test[i] -
                              Rate[int(X_test[i, 0]), int(X_test[i, 1])])
        # 573188(10),51992(20),46243(40),36958(100)
        print("在测试集上的总分误差：{0}s".format(sum1))
        RMSE, MAE = self.Get_RMSEandMAE(Rate, X_test, y_test)
        # 2.29(10)，2.13(20),1.93(40),1.57(100)
        print("测试集上的回归误差RSE：{0}".format(RMSE))
        # MAE = self.getMAE(Rate,X_test,y_test)
        print("测试集上的MAE：{0}".format(MAE))

def ReadMysql(host,port,username,password,database):
    db = pymysql.connect(host = host,user = username,password = password,database = database,port = port, charset='utf8mb4')
    cursor = db.cursor()
    cursor.execute("select userID_id, movieID_id, rating from Rating_rating")
    results = cursor.fetchall()
    userID, movieID, rating = [],[],[]
    for item in results:
        userID.append(item[0])
        movieID.append(item[1])
        rating.append(item[2])

    return csr_matrix( (np.array(rating),( np.array(userID), np.array(movieID) ) ), shape=(6500,4500) )#shape = (6500,4500))

def Train():
    Configuration = {
        'host': "gz-cdb-5clnvoq5.sql.tencentcdb.com",
        'port': 60720,
        'username': "root",
        'password': "MySqlxmhllly@17227016",
        'database': "db_for_project"
    }
    lfm = LFM(lfm_num=20)
    sparse_matrix = ReadMysql(Configuration['host'], Configuration['port'], Configuration['username'], 
                            Configuration['password'], Configuration['database'])
    X_train,X_test,y_train,y_test = lfm.Fit(sparse_matrix)
    lfm.Train(X_train,y_train, alpha = 0.005,lambda_u = 0.1,lambda_i = 0.12,train_times = 100)

    lfm_info = [lfm.lfm_num, lfm.alpha, lfm.lambda_u,
          lfm.lambda_i, lfm.r_max, lfm.c_max]
    output_file = open('/home/UndergraduateMovieRecommand/Recommend/lfm_info.pkl', 'wb')
    lfm_str = pickle.dumps(lfm_info)
    output_file.write(lfm_str)
    output_file.close()

    output_file = open('/home/UndergraduateMovieRecommand/Recommend/lfm_Up.pkl', 'wb')
    lfm_str = pickle.dumps(lfm.Up)
    output_file.write(lfm_str)
    output_file.close()

    output_file = open('/home/UndergraduateMovieRecommand/Recommend/lfm_VTp.pkl', 'wb')
    lfm_str = pickle.dumps(lfm.VTp)
    output_file.write(lfm_str)
    output_file.close()

    
def test():
    

    lfm = LFM(lfm_num=10)  # lfm_num 设置模型隐向量的维度
    #如果之前训练的模型已经存在，则直接读取文件，恢复模型
    # print(os.path.abspath(__file__))
    # print("nooo")
    try:
        with open(r'./lfm_sql.pkl','rb') as f:
            lfm = pickle.loads(f.read())
            print("读取成功")
    except IOError:
        print("File not exist!")
    # return 
    #给出scipy稀疏矩阵的存储文件的路径，传入路径返回训练和测试数据集
    # npz_path = r'E:\MyProject\Recommend_code_origin\sparse_matrix_100k.npz'
    host = "gz-cdb-5clnvoq5.sql.tencentcdb.com"
    username = "root"
    password = "MySqlxmhllly@17227016"
    database = "db_for_project"
    port = 60720
    sparse_matrix = ReadMysql(host,port,username,password,database)
    X_train,X_test,y_train,y_test = lfm.Fit(sparse_matrix)

    #模型的训练    
    lfm.Train(X_train,y_train, alpha = 0.005,lambda_u = 0.1,lambda_i = 0.12,train_times = 2)
    #1m: 0.001 0.2 0.2 10
    #100k: 0.07 0.1 0.12 100  
    #模型的测试
    lfm.lfm_test(X_test,y_test)
    #将训练号的模型存储下来
    output_file = open('lfm_sql.pkl','wb')
    lfm_str = pickle.dumps(lfm)
    output_file.write(lfm_str)
    output_file.close()

if __name__ == '__main__':
    test()



