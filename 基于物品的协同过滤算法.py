import numpy as np
import pandas as pd
import random


# 读取数据集
data_df = pd.read_csv('./ml-100k/u.data', header=None, index_col=None)
data = data_df.values




# 划分数据集
def split_data(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for d in data:
        if random.randint(0, M) == k:
            test.append(d)
        else:
            train.append(d)
    return train, test

train_df, test_df = split_data(data, 8, 2, 2)



:
# 整理数据集   943 users on 1682 movies.
# 矩阵：横坐标表示对应一个用户给一个电影打的分数， 纵坐标表示用户id   
# train  
train_data = np.zeros((944, 1683), dtype=np.int)   
for data in train_df:
    infor = data[0].split('\t')
    user = int(infor[0])
    movie = int(infor[1])
    train_data[user][movie] = int(infor[2])
train_data = np.transpose(train_data)
test_data = np.zeros((944, 1683))   
for data in test_df:
    infor = data[0].split('\t')
    user = int(infor[0])
    movie = int(infor[1])
    test_data[user][movie] = int(infor[2])
test_data = np.transpose(test_data)



def cos_sim(x, y):
    """余弦相似性

    Args:
    - x: mat, 以行向量的形式存储
    - y: mat, 以行向量的形式存储

    :return: x 和 y 之间的余弦相似度
    """
    numerator = np.matmul(x, y.T)  # x 和 y 之间的内积
    denominator = np.sqrt(np.matmul(x, x.T)) * np.sqrt(np.matmul(y, y.T))
    return (numerator / max(denominator, 1e-7))



# 对于任意矩阵，计算任意两个行向量之间的相似度：
def similarity(data):
    m = np.shape(data)[0]  # 用户的数量
    # 初始化相似矩阵
    w = np.zeros((m, m))  # 相似度矩阵w是一个对称矩阵，而且在相似度矩阵中，约定自身的相似度的值为 $0$ 

    for i in range(m):
        for j in range(i, m):
            if not j == i:
                # 计算任意两行之间的相似度
                w[i, j] = cos_sim(data[i], data[j])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w

def item_based_recommend(data, w, user):
    m, n = np.shape(data)  # m: 物品数量 n: 用户数量
    interaction = data[:, user].T  # 用户 user 互动物品信息

    # 找到用户 user 没有互动的商品
    not_iter = []
    for i in range(m):
        if interaction[i] == 0:  # 用户 user 未打分项
            not_iter.append(i)

    # 对没有互动过的物品进行预测
    predict = {}
    for x in not_iter:
        item = np.copy(interaction)  # 获取用户 user 对物品的互动信息
        for j in range(m):   # 对每一个物品
            if item[j] != 0:  # 利用互动过的物品预测
                if x not in predict:
                    predict[x] = w[x][j] * item[j]
                else:
                    predict[x] = predict[x] + w[x][j] * item[j]
    # 按照预测的大小从大到小排序
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)



def top_k(predict, n):
    """为用户推荐前 n 个物品

    Args:
    - predict: list, 排好序的物品列表
    - k: int, 推荐的物品个数

    :return: top_recom, list, top n 个物品
    """
    top_recom = []
    len_result = len(predict)
    if n >= len_result:
        top_recom = predict
    else:
        for i in range(n):
            top_recom.append(predict[i][0])
    return top_recom





#  准确率
def prediction(train_data, test_data, similarity, n):
    pre = 0.0
    for i in range(100):   # 随机选取100个用户
        succ = 0
        user_id = random.randint(0, 943) 
        predict_all = item_based_recommend(train_data, similarity, user_id)
        predict = top_k(predict_all, n)
        test = np.where(test_data[user_id]!=0)
        for j in range(n):
            if predict[j] in test[0]:
                succ += 1
        pred = succ/n
        pre = pre + pred
        
        
    pre = pre / 100
    
    return pre

similarity = similarity(train_data)
pre = prediction(train_data, test_data, similarity, 10)
print(pre)




# 召回率
def recall(train_data, test_data, similarity, n):
    pre = 0.0
    for i in range(100):   # 随机选取100个用户
        succ = 0
        user_id = random.randint(0, 944) 
        predict_all = item_based_recommend(train_data, similarity, user_id)
        predict = top_k(predict_all, n)
        test = np.where(test_data[user_id]!=0)
        if len(predict) < n:
            n = len(predict)
        for j in range(n):
            if predict[j] in test[0]:
                succ += 1
        pred = succ/max(test[0].shape[0],1e-7)
        pre = pre + pred
        
        
    pre = pre / 100
    
    return pre

similarity = similarity(train_data)
pre = recall(train_data, test_data, similarity, 10)
print(pre)


# 覆盖率

def coverage(train_data, test_data, similarity, n):
    recommond_item = set()
    all_item = set()
    for i in range(100):   # 随机选取100个用户
        user_id = random.randint(0, 943) 
        predict_all = item_based_recommend(train_data, similarity, user_id)
        predict = top_k(predict_all, n)
        test = np.where(test_data[user_id]!=0)
        test = test[0]
        train = np.where(train_data[user_id]!=0)
        train = train[0]
        for tra in train:
            all_item.add(tra)
        for j in range(n):
            if predict[j] in test:
                recommond_item.add(predict[j])
        
        pre = len(recommond_item)/len(all_item)

    
    return pre

similarity = similarity(train_data)
pre = coverage(train_data, test_data, similarity, 10)
print(pre)



# 新颖度

import math

def popularity(train, test, similarity, N):
    item_popularity = dict()
    
    for i in range(100):   # 随机选取100个用户
        user_id = random.randint(0, 943) 
        predict_all = item_based_recommend(train_data, similarity, user_id)
        train = np.where(train_data[user_id]!=0)
        train = train[0]
        for item in train:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
        ret = 0
        n = 0
        rank = top_k(predict_all, N)
        for item in rank:
            ret0 = item_popularity.get(item, 0)
            ret += math.log(1 + ret0)
            n += 1
    ret /= n * 1.0
    return ret

similarity = similarity(train_data)
pre = popularity(train_data, test_data, similarity, 10)
print(pre)
