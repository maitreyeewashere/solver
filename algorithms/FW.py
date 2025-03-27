def floydwarshall(n, G):
    # Create a copy of the input matrix
    distance = [row[:] for row in G]
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][k] != float('inf') and distance[k][j] != float('inf'):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    
    # Replace infinity values with '∞' for display
    for i in range(n):
        for j in range(n):
            if distance[i][j] == float('inf'):
                distance[i][j] = '∞'
            elif distance[i][j] >= 999:  # Handle large numbers that represent infinity
                distance[i][j] = '∞'
    
    return distance