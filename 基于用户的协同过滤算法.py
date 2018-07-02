import numpy as np
import math
import random

def load_movielens(path="./ml-100k"):

    train = {}
    test = {}

    for line in open(path + '/ua.base', encoding='latin-1'):
        user, movieid, rating, ts = line.split('\t')
        train.setdefault(user, {})
        train[user][movieid] = float(rating)

    for line in open(path + '/ua.test', encoding='latin-1'):
        user, movieid, rating, ts = line.split('\t')
        test.setdefault(user, {})
        test[user][movieid] = float(rating)

    user_counts = 0
    movie_counts = 0
    for line in open(path + "/u.info", encoding='latin-1'):

        count, content = line.strip().split(" ")
        if "users" in content:
            user_counts = int(count)
        elif "items" in content:
            movie_counts = int(count)
    return train, test, user_counts, movie_counts



# 计算召回率
def recall(train, test, N, rank):
    hit = 0
    all = 0
    new_rank = top_k(rank, N)
    for user in train.keys():
        tu = test[user]
        for item, pui in new_rank:
            if str(item+1) in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)


# 计算准确率
def precision(train, test, N, rank):
    hit = 0
    all = 0
    new_rank = top_k(rank, N)
    for user in train.keys():
        tu = test[user]
        for item, pui in new_rank:
            if str(item+1) in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)


# 计算覆盖率
def coverage(train, test, N, rank):
    recommend_items = set()
    all_items = set()
    new_rank = top_k(rank, N)
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)

        for item, pui in new_rank:
            recommend_items.add(str(item+1))
    return len(recommend_items) / (len(all_items) * 1.0)


# 平均流行度
def popularity(train, test, N, rank):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        new_rank = top_k(rank, N)
        for item, pui in new_rank:
            ret += math.log(1 + item_popularity[str(item+1)])
            n += 1
    ret /= n * 1.0
    return ret



def data2mat(data, user_counts=943, movie_counts=1682):
    # user_counts = len(data)
    # print("user_counts = ", user_counts)
    # movie_counts = len(set(sum([list(x.keys()) for x in data.values()], [])))
    # print("movie_counts = ", movie_counts)
    mat = np.zeros((user_counts, movie_counts), dtype=float)
    for user, movies in data.items():
        for movie, score in movies.items():
            mat[int(user)-1][int(movie)-1] = float(int(score))

    return mat



def cos_sim(x, y):
    numerator = np.matmul(x, y.T)  # x 和 y 之间的内积
    denominator = np.sqrt(np.matmul(x, x.T)) * np.sqrt(np.matmul(y, y.T))
    return (numerator / denominator)



def similarity(data):
    m = np.shape(data)[0]  # 用户的数量
    # 初始化相似矩阵
    w = np.mat(np.zeros((m, m)))

    for i in range(m):
        for j in range(i, m):
            if not j == i:
                # 计算任意两行之间的相似度
                w[i, j] = cos_sim(data[i], data[j])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w



def user_based_recommend(data, w, user):
    m, n = np.shape(data)
    interaction = data[int(user)-1,]  # 用户 user 与物品信息
    
    # 找到用户 user 没有互动过的物品
    not_inter = []
    for i in range(n):

        if interaction[i] == 0:  # 没有互动的物品
            not_inter.append(i)
    # print(not_inter)
    # 对没有互动过的物品进行预测
    predict = {}
    for x in not_inter:
        item = np.copy(data[:, x])  # 找到所有用户对商品 x 的互动信息

        for i in range(m):  # 对每一个用户
            if item[i] != 0:
                if x not in predict:
                    predict[x] = w[user, i] * item[i]
                else:
                    predict[x] = predict[x] + w[user, i] + item[i]
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)



def top_k(predict, n):

    top_recom = []
    len_result = len(predict)
    if n >= len_result:
        top_recom = predict
    else:
        for i in range(n):
            top_recom.append(predict[i])
    return top_recom



def main():
    train, test, user_counts, movie_counts = load_movielens()
    # print(prefs['1'])
    print("用户数量:", user_counts, "电影数量：", movie_counts)
    # print(train[0],'\n',test[0])
    mat = data2mat(train)
    w = similarity(mat)
    rank = user_based_recommend(mat,w, 90)

    print(top_k(rank, 10))
    print("召回率：",recall(train, test, 10, rank))
    print("准确率：", precision(train, test, 10, rank))
    print("覆盖率：", coverage(train, test, 10, rank))
    print("流行度：", popularity(train, test, 10, rank))



if __name__ == '__main__':
