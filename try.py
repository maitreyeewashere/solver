import matplotlib.pyplot as plt

def generate_blank_graph():
    plt.figure(figsize=(6, 6))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means Clustering")
    plt.grid(True)

    # Save the blank image
    plt.savefig("static/clusters.png")
    plt.close()

# Call the function to generate the blank graph
generate_blank_graph()
