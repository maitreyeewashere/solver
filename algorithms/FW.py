from flask import Flask, request, jsonify

app = Flask(__name__)

def find_shortest_paths(graph):
    num_vertices = len(graph)
    distance_matrix = [[x for x in row] for row in graph]    
    for intermediate_vertex in range(num_vertices):
        for start_vertex in range(num_vertices):
            for end_vertex in range(num_vertices):
                distance_matrix[start_vertex][end_vertex] = min(
                    distance_matrix[start_vertex][end_vertex],
                    distance_matrix[start_vertex][intermediate_vertex] + distance_matrix[intermediate_vertex][end_vertex]
                )
    return distance_matrix
@app.route('/floydwarshall', methods=['POST'])
def calculate_shortest_paths():
    graph = request.json['graph']
    shortest_paths = find_shortest_paths(graph)
    return jsonify(shortest_paths)
if __name__ == '__main__':
    app.run(debug=True)