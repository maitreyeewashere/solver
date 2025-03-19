from flask import Flask, request, jsonify
import heapq
from collections import defaultdict

app = Flask(__name__)

class Node:
    def __init__(self, character, frequency):
        self.character = character
        self.frequency = frequency
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.frequency < other.frequency
def calculate_frequencies(message):
    frequency_dict = defaultdict(int)
    for char in message:
        frequency_dict[char] += 1
    return frequency_dict
def build_huffman_tree(frequency_dict):
    priority_queue = [Node(char, freq) for char, freq in frequency_dict.items()]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        node1 = heapq.heappop(priority_queue)
        node2 = heapq.heappop(priority_queue)
        merged_node = Node(None, node1.frequency + node2.frequency)
        merged_node.left = node1
        merged_node.right = node2
        heapq.heappush(priority_queue, merged_node)
    return priority_queue[0]
def generate_codes(root, current_code, codes):
    if root is None:
        return
    if root.character is not None:
        codes[root.character] = current_code
    generate_codes(root.left, current_code + "0", codes)
    generate_codes(root.right, current_code + "1", codes)
def encode_message(message):
    frequency_dict = calculate_frequencies(message)
    huffman_tree = build_huffman_tree(frequency_dict)
    codes = {}
    generate_codes(huffman_tree, "", codes)
    encoded_message = "".join(codes[char] for char in message)
    return encoded_message, codes
@app.route('/huffman/encode', methods=['POST'])
def encode_with_huffman():
    message = request.json['message']
    encoded_message, codes = encode_message(message)
    return jsonify({'encoded_message': encoded_message, 'codes': codes})
if __name__ == '__main__':
    app.run(debug=True)