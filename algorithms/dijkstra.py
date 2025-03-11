import heapq
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os

def dijkstra(adj_list, start,end):
    graph = nx.DiGraph()

    for node, neighbours in adj_list.items():
        for neighbour, weight in neighbours:
            graph.add_edge(node, neighbour, weight=weight)

    pq = [(0,start)] #minheap (cost, node)
    distances = {node: float('inf') for node in graph.nodes} #setting initial distances to infinity
    distances[start] = 0

    parent = {node: None for node in graph.nodes}

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)

        if curr_node == end:
            break

        for neighbour in graph.neighbors(curr_node):
            weight = graph[curr_node][neighbour]['weight']
            new_dist = curr_dist + weight

            if new_dist < distances[neighbour]:
                distances[neighbour] = new_dist
                parent[neighbour] = curr_node
                heapq.heappush(pq,(new_dist,neighbour))


    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, distances[end], graph

def visualiseGraph(adj_list,path):
    graph = nx.DiGraph()

    for node, neighbours in adj_list.items():
        for neighbour, weight in neighbours:
            graph.add_edge(node, neighbour, weight=weight)

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8,6))
    nx.draw(graph, pos,with_labels=True,node_color='lightblue',edge_color='gray',node_size=2000,font_size=10)

    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="red", width=2.5)

    edge_labels = {(u, v): f"{graph[u][v]['weight']}" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    img_path = "static/graph.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

'''
adj_list = {
    "A": [("B", 4), ("C", 1)],
    "B": [("D", 5)],
    "C": [("B", 2), ("D", 8)],
    "D": []
}

start = "A"
end = "D"


path, cost, graph = dijkstra(adj_list, start, end)
print(f"Shortest path: {path}")
print(f"Total cost: {cost}")

img_path = visualiseGraph(adj_list, path)
print(f"Graph saved at: {img_path}")'''
