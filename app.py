from flask import *
from algorithms.greedy import *
from algorithms.dijkstra import *
from algorithms.linreg import *
from algorithms.dp import *
from algorithms.KMeans import kmeans
from algorithms.FW import *
import os
import json
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/greedy/fractional-knapsack', methods=['GET', 'POST'])
def frac_knapsack():
    result = None
    if request.method == 'POST':
        cap = float(request.form['capacity'])
        profits = list(map(float, request.form.getlist('profits')))
        weights = list(map(float, request.form.getlist('weights')))
        if len(profits) != len(weights):
            return "Mismatched profits/weights count!", 400
        if any(w <= 0 for w in weights):
            return "Weights must be positive!", 400
        result = fracKnapsack(cap, list(zip(profits, weights)))
    
    return render_template('fracknap.html', result=result)

@app.route('/greedy/job-sequencing', methods=['GET', 'POST'])
def job_sequence():
    result = None
    if request.method == 'POST':
        profits = list(map(float, request.form.getlist('profits')))
        deadlines = list(map(int, request.form.getlist('deadlines')))
        result = jobSequence(list(zip(profits, deadlines)))
    
    return render_template('jobs.html', result=result)

@app.route('/dp/knapsack', methods=['GET', 'POST'])
def binary_knapsack():
    result = None
    if request.method == 'POST':
        cap = float(request.form['capacity'])
        profits = list(map(float, request.form.getlist('profits')))
        weights = list(map(float, request.form.getlist('weights')))
        result = knapsack01(cap, zip(profits, weights))
    
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
        start = request.form["start"].upper()
        end = request.form["end"].upper()
        
        try:
            adj_list = json.loads(request.form["adj_list"])
            if start not in adj_list or end not in adj_list:
                return render_template("dijkstra.html", error="Start/end node not in graph")
        except json.JSONDecodeError:
            return render_template("dijkstra.html", error="Invalid adjacency list format.")
        
        path, cost, _ = dijkstra(adj_list, start, end)
        
        if cost == float('inf'):
            return render_template("dijkstra.html", error="No path exists between nodes")
        
        img_path = visualiseGraph(adj_list, path)
        
        return render_template("dijkstra.html", path=path, cost=cost, img_path=img_path)
    
    return render_template("dijkstra.html")

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/graph/floyd-warshall', methods=["GET", "POST"])
def floyd_warshall():
    matrix = None
    result = None  
    size = None  # Initialize size

    if request.method == 'POST':
        size = int(request.form.get('size', 0))  # Get size safely

        # Create adjacency matrix from form input
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                value = request.form.get(f'matrix-{i}-{j}')  
                row.append(int(value))
            matrix.append(row)

        result = floydwarshall(size, matrix) 
        
        print('result',result) # Run 

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
        xlist = request.form.getlist('x', type=float)
        ylist = request.form.getlist('y', type=float)
        
        if xlist and ylist:
            b = regression(xlist, ylist)
            result = f'y = {b[1]:.2f}x + {b[0]:.2f}'
            save_plot(xlist, ylist, b)
    else:
        save_plot()
    
    return render_template('linreg.html', result=result, img_path=f'/{img_path}')

@app.route('/ml/kmeans', methods=['GET', 'POST'])
def handle_kmeans():
    if request.method == 'POST':
        try:
            # Get form data
            k = int(request.form['k'])
            data_str = request.form['list']
            
            # Convert string input to numpy array
            data = np.array(ast.literal_eval(data_str))
            
            # Run K-Means algorithm
            clusters = kmeans(data, k=k)
            
            # Create visualization
            plt.figure(figsize=(8, 6))
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            
            for i, cluster in enumerate(clusters):
                if cluster['points']:
                    points = np.array(cluster['points'])
                    plt.scatter(points[:,0], points[:,1], 
                               c=colors[i%len(colors)], 
                               label=f'Cluster {i+1}')
                    plt.scatter(cluster['center'][0], cluster['center'][1],
                               marker='x', s=200, c='black', linewidths=3)
            
            plt.legend()
            plt.grid(True)
            
            # Save plot with cache-busting timestamp
            timestamp = str(int(time.time()))
            plot_path = os.path.join(app.config['STATIC_FOLDER'], f'clusters_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            
            return render_template('kmeans.html', 
                                 plot_url=f'clusters_{timestamp}.png')
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    # Initial GET request
    return render_template('kmeans.html', plot_url=None)

if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=5000, debug=True)
    #print(app.url_map)
