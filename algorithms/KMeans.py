from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/kmeans/predict', methods=['POST'])
def predict_clusters():
    data = request.json['data']
    num_clusters = request.json['num_clusters']
    data_array = np.array(data).reshape(-1, len(data[0]))
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(data_array)
    predicted_clusters = kmeans_model.predict(data_array)
    return jsonify({'predicted_clusters': predicted_clusters.tolist()})
if __name__ == '__main__':
    app.run(debug=True)
