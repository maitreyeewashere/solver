from flask import *
from algorithms.greedy import *
from algorithms.dijkstra import *
from algorithms.linreg import *
from algorithms.dp import *
from algorithms.KMeans import *
from algorithms.FW import *
import os
import json
import matplotlib
import io
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import ast

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/greedy/fractional-knapsack', methods=['GET', 'POST'])
def frac_knapsack():
    result = None
    if request.method == 'POST':
        try:
            cap = float(request.form['capacity'])
            profits = list(map(float, request.form.getlist('profits')))
            weights = list(map(float, request.form.getlist('weights')))
            
            if len(profits) != len(weights):
                return "Mismatched profits/weights count!", 400
            if any(w <= 0 for w in weights):
                return "Weights must be positive!", 400
            
            result = fracKnapsack(cap, list(zip(profits, weights)))
        except ValueError:
            return "Invalid input! Please enter valid numbers.", 400
    
    return render_template('fracknap.html', result=result)

@app.route('/greedy/job-sequencing', methods=['GET', 'POST'])
def job_sequence():
    result = None
    if request.method == 'POST':
        try:
            profits = list(map(float, request.form.getlist('profits')))
            deadlines = list(map(int, request.form.getlist('deadlines')))
            result = jobSequence(list(zip(profits, deadlines)))
        except ValueError:
            return "Invalid input! Please enter valid numbers.", 400
    
    return render_template('jobs.html', result=result)

@app.route('/dp/knapsack', methods=['GET', 'POST'])
def binary_knapsack():
    result = None
    if request.method == 'POST':
        try:
            cap = float(request.form['capacity'])
            profits = list(map(float, request.form.getlist('profits')))
            weights = list(map(float, request.form.getlist('weights')))
            result = knapsack01(cap, list(zip(profits, weights)))
        except ValueError:
            return "Invalid input! Please enter valid numbers.", 400
    
    return render_template('knapsack.html', result=result or {"length": None, "dp_table": None})

@app.route('/dp/lcs', methods=['GET', 'POST'])
def longest_cs():
    result = None 
    if request.method == 'POST':
        str1 = request.form.get('string1', '').strip()
        str2 = request.form.get('string2', '').strip()
        if not str1 or not str2:
            result = {"error": "Both strings are required!"}  
        else:
            result = lcs(str1, str2)

    return render_template('lcs.html', result=result)

@app.route('/dp/mcm', methods=['GET', 'POST'])
def matrix_chain():
    try:
        dimensions = list(map(int, request.form.getlist("dimensions")))
        if len(dimensions) < 2:
            return render_template("mcm.html", error="Please enter at least two dimensions.")

        result = mcm(dimensions)
        return render_template("mcm.html", result={"cost": result["cost"], "dp_table": result["dp_table"]})
    
    except Exception as e:
        return render_template("mcm.html", error=str(e))

@app.route('/graph/dijkstra', methods=["GET", "POST"])
def dijkstra_solver():
    if request.method == "POST":
        try:
            start = request.form["start"].upper()
            end = request.form["end"].upper()
            adj_list = json.loads(request.form["adj_list"])
            
            if start not in adj_list or end not in adj_list:
                return render_template("dijkstra.html", error="Start/end node not in graph")

            path, cost, _ = dijkstra(adj_list, start, end)
            if cost == float('inf'):
                return render_template("dijkstra.html", error="No path exists between nodes")

            img_path = visualiseGraph(adj_list, path)
            return render_template("dijkstra.html", path=path, cost=cost, img_path=img_path)
        except json.JSONDecodeError:
            return render_template("dijkstra.html", error="Invalid adjacency list format.")
    
    return render_template("dijkstra.html")

@app.route('/graph/floyd-warshall', methods=["GET", "POST"])
def floyd_warshall():
    matrix = None
    result = None  
    size = None  

    if request.method == 'POST':
        try:
            size = int(request.form.get('size', 0))
            matrix = [[int(request.form.get(f'matrix-{i}-{j}', 0)) for j in range(size)] for i in range(size)]
            result = floydwarshall(size, matrix) 
        except ValueError:
            return render_template("floyd.html", error="Invalid input! Please enter numbers only.")
    
    return render_template('floyd.html', matrix=matrix, result=result, size=size)

@app.route('/ml/linear-regression', methods=['GET', 'POST'])
def linear_regression():
    img_path = "static/plot.png"
    result = None
    
    def save_plot(xlist=None, ylist=None, line_params=None):
        plt.figure(figsize=(6, 4), dpi=100)
        plt.grid(True)
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        
        if xlist and ylist:
            plt.scatter(xlist, ylist, color='blue', label='Data Points')
            if line_params:
                plt.plot(np.array(xlist), line_params[0] + line_params[1] * np.array(xlist),
                         color='red', label=f'Line: y = {line_params[1]:.2f}x + {line_params[0]:.2f}')
        
        plt.legend()
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
    
    if request.method == 'POST':
        try:
            xlist = list(map(float, request.form.getlist('x')))
            ylist = list(map(float, request.form.getlist('y')))
            b = regression(xlist, ylist)
            result = f'y = {b[1]:.2f}x + {b[0]:.2f}'
            save_plot(xlist, ylist, b)
        except ValueError:
            return "Invalid input! Please enter valid numbers.", 400
    else:
        save_plot()
    
    return render_template('linreg.html', result=result, img_path=f'/{img_path}')


@app.route('/ml/kmeans', methods=['POST', 'GET'])
def run_kmeans():
    if request.method == 'POST':
        try:

            k = int(request.form['k'])
            data_str = request.form['list'].strip()


            try:
                X = np.array(json.loads(data_str))
            except json.JSONDecodeError:
                return "Error: Invalid input format. Use JSON-style format: [[1,2], [3,4], [4,5]]"


            clusters = kmeans(X, k)

            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'pink']

            for i, cluster in enumerate(clusters):
                points = np.array(cluster['points'])
                
                if points.shape[0] > 0:

                    plt.scatter(points[:, 0], points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

                    distances = [distance(np.array(point), np.array(cluster['center'])) for point in points]
                    radius = max(distances) if distances else 0
   
                    circle = plt.Circle(cluster['center'], radius, color=colors[i % len(colors)], fill=True, alpha=0.8, linestyle='dashed', linewidth=2)
                    ax.add_patch(circle)

                plt.scatter(cluster['center'][0], cluster['center'][1], color='black', marker='x', s=100)

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.title('K-Means Clusters')
            plt.legend()

            img_path = 'static/clusters.png'
            plt.savefig(img_path)
            plt.close()

            return render_template('kmeans.html', result=clusters, img_path=img_path)

        except Exception as e:
            return f"Error: {str(e)}"

    blank_img_path = generate_blank_graph()
    return render_template('kmeans.html', result=None, img_path=blank_img_path)

    
    

if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
