from flask import Flask, render_template, request
from algorithms.greedy import fracKnapsack, jobSequence
from algorithms.dijkstra import dijkstra, visualiseGraph
from algorithms.linreg import regression
import os
import json
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
        result = fracKnapsack(cap, zip(profits, weights))
    
    return render_template('fracknap.html', result=result)


@app.route('/greedy/job-sequencing', methods=['GET', 'POST'])
def job_sequence():
    result = None
    if request.method == 'POST':
        profits = list(map(float, request.form.getlist('profits')))
        deadlines = list(map(int, request.form.getlist('deadlines')))
        result = jobSequence(zip(profits, deadlines))
    
    return render_template('jobs.html', result=result)


@app.route('/graph/dijkstra', methods=["GET", "POST"])
def dijkstra_solver():
    if request.method == "POST":
        start, end = request.form["start"], request.form["end"]
        try:
            adj_list = json.loads(request.form["adj_list"])
        except json.JSONDecodeError:
            return render_template("dijkstra.html", error="Invalid adjacency list format.")
        
        path, cost, _ = dijkstra(adj_list, start, end)
        img_path = visualiseGraph(adj_list, path)
        return render_template("dijkstra.html", path=path, cost=cost, img_path=img_path)
    
    return render_template("dijkstra.html")


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

if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
