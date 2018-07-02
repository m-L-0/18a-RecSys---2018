import sys
import random
import math
import numpy as np

def load_movielens(path='../ml-100k'):
    # get movie titles
    movies = {}
    for line in open(path + '/u.item', encoding='latin-1'):
        id, title = line.split('|')[0:2]
        movies[id] = title
    # load data

    train = {}

    for line in open(path + '/ua.base', encoding='latin-1'):
        user, movieid, rating, ts = line.split('\t')
        train.setdefault(user, {})
        train[user][movieid] = float(rating)
    test = {}
    for line in open(path + '/ua.test', encoding='latin-1'):
        user, movieid, rating, ts = line.split('\t')
        test.setdefault(user, {})
        test[user][movieid] = float(rating)
    return train, test, movies


def gen_list(data, user_len=943, movie_len=1682):
    mat_data = np.zeros((user_len, movie_len), dtype=float)
    for u, item in data.items():
        for n, r in item.items():
            mat_data[int(u) - 1][int(n) - 1] = float(int(r))
    return mat_data


def sgd(data_matrix, k, alpha, lam, max_cycles):
    m, n = np.shape(data_matrix)
    p = np.mat(np.random.random((m, k)))
    q = np.mat(np.random.random((k, n)))
    # start training
    for step in range(max_cycles):
        for i in range(m):
            for j in range(n):
                if data_matrix[i, j] > 0:
                    error = data_matrix[i, j]
                    for r in range(k):
                        error = error - p[i, r] * q[r, j]
                    for r in range(k):
                        p[i, r] = p[i, r] + alpha * (
                            2 * error * q[r, j] - lam * p[i, r])
                        q[r, j] = q[r, j] + alpha * (
                            2 * error * p[i, r] - lam * q[r, j])
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if data_matrix[i, j] > 0:
                    error = 0.0
                    for r in range(k):
                        error = error + p[i, r] * q[r, j]
                    # calculate loss function
                    loss = (data_matrix[i, j] - error) * (
                        data_matrix[i, j] - error)
                    for r in range(k):
                        loss = loss + lam * (
                            p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2
    return p, q



def prediction(data_matrix, p, q, user):

    n = np.shape(data_matrix)[1]
    predict = {}
    for j in range(n):
        if data_matrix[int(user) - 1, j] == 0:
            predict[j] = (p[int(user) - 1, ] * q[:, j])[0, 0]

    # 按照打分从大到小排序
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)



def top_k(rank, k):
    if len(rank) <= k:
        return rank
    else:
        return rank[:k]
    


def recall(train, test, W, N, K):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = top_k(prediction(mat_data, N, K, user), W)
        for item, pui in rank:
            if str(item + 1) in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)



def precision(train, test, W, N, K):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = top_k(prediction(mat_data, N, K, user), W)
        for item, pui in rank:
            if str(item + 1) in tu:
                hit += 1
        all += W
    return hit / (all * 1.0)





def coverage(train, test, W, N, K):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = top_k(prediction(mat_data, N, K, user), W)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)





def popularity(train, test, W, N, K):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            item_popularity[item] = item_popularity.get(item, 0) + 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = top_k(prediction(mat_data, N, K, user), W)
        for item, pui in rank:
            ret += math.log(1 + 1 / item_popularity[str(item + 1)])
            n += 1
    ret /= n * 1.0
    return ret



if __name__ == '__main__':
    print("Loadind dataset...")
    train, test, movies = load_movielens()
    mat_data = gen_list(train)
    print("Start training")
    p, q = sgd(mat_data, 5, 0.001, 0.01, 50)
    rank = prediction(mat_data, p, q, 1)

    recall = recall(train, test, 5, p, q)
    precision = precision(train, test, 5, p, q)
    popularity = popularity(train, test, 5, p, q)
    coverage = coverage(train, test, 5, p, q)
    print('recall: ', recall, '\n')
    print('precision: ', precision, '\n')
    print('Popularity: ', popularity, '\n')
    print('coverage: ', coverage, '\n')
else:
    print("this is not the main function")
