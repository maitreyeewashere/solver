from flask import Flask, render_template, request
from algorithms.greedy import fracKnapsack, jobSequence

app = Flask(__name__)

# Main Menu
@app.route('/')
def index():
    return render_template('main.html')


@app.route('/greedy/fractional-knapsack')
def frac_knapsack_page():
    return render_template('fracknap.html')

@app.route('/greedy/frac_knapsack', methods=['POST'])
def frac_knapsack_solver():
    result = None
    if request.method == 'POST':
        cap = float(request.form['capacity'])
        profits = [float(p) for p in request.form.getlist('profits')]
        weights = [float(w) for w in request.form.getlist('weights')]
        items = list(zip(profits, weights))  
        result = fracKnapsack(cap, items)

    return render_template('fracknap.html', result=result)


@app.route('/greedy/job-sequencing')
def job_sequence_page():
    return render_template('jobs.html',result=None)

@app.route('/greedy/jobs', methods=['POST'])
def job_sequence_solver():
    result = None
    if request.method == 'POST':
        profits = [float(p) for p in request.form.getlist('profits')]
        deadlines = [int(d) for d in request.form.getlist('deadlines')]
        jobs = list(zip(profits, deadlines)) 
        result = jobSequence(jobs)

    return render_template('jobs.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
