def floydwarshall(n, G):
    # Create a copy of the input matrix
    distance = [row[:] for row in G]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][k] != float('inf') and distance[k][j] != float('inf'):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    for i in range(n):
        for j in range(n):
            if distance[i][j] == float('inf'):
                distance[i][j] = '∞'
            elif distance[i][j] >= 999:
                distance[i][j] = '∞'
    
    return distance
