import numpy as np
import matplotlib.pyplot as plt

def distance(x, y):
    return np.linalg.norm(x - y)

def assign(k, X, clusters):
    for i in range(k):  
        clusters[i]['points'] = []  

    for i in range(X.shape[0]):
        dist = [distance(X[i], clusters[j]['center']) for j in range(k)]
        closest_cluster = np.argmin(dist)
        clusters[closest_cluster]['points'].append(X[i])

def update(k, clusters):
    for i in range(k):
        if clusters[i]['points']:  
            clusters[i]['center'] = np.mean(clusters[i]['points'], axis=0)

def predict(X, k, clusters):
    return [np.argmin([distance(x, clusters[j]['center']) for j in range(k)]) for x in X]

def kmeans(X, k=3, max_iters=10):
    np.random.seed(23)  
    clusters = {i: {'center': X[np.random.randint(0, X.shape[0])], 'points': []} for i in range(k)}

    for _ in range(max_iters):
        assign(k, X, clusters)
        update(k, clusters)

    return list(clusters.values())  # Return clusters properly

#print(kmeans(np.array([(1,2),(3,4),(5,6),(7,8)]),3))
