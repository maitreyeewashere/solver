def floydwarshall(n, G):
    distance = [row[:] for row in G]


    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    for i in range(n):
        for j in range(n):
            if (distance[i][j] >= 999):
                distance[i][j] = '∞'
               

    return distance
