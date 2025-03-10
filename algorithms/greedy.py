import heapq

def fracKnapsack(cap, arr):
    #arr[0][0] = profit, arr[0][1] = weight
    arr.sort(key = lambda x: x[0]/x[1], reverse = True)
    finalvalue = 0.0
    select = []

    for item in arr:
        if item[1] <= cap:
            cap -= item[1]
            finalvalue += item[0]
            select.append(1)

        elif cap != 0:
            finalvalue += item[0] * cap / item[1]
            select.append(cap/item[1])
            break

        else:
            select.append(0)


    return (finalvalue,select)

def jobSequence(jobs):
    import heapq  # Ensure heapq is imported
    
    # jobs[i] = (profit, deadline)
    jobs.sort(key=lambda x: (-x[0], x[1]))  # Sort by profit DESC, deadline ASC

    pq = []  # Min heap for tracking scheduled jobs
    total_profit = 0
    jobs_scheduled = []

    for i,(profit, deadline) in enumerate(jobs):
        if len(pq) < deadline:  # If we can schedule it within the deadline
            heapq.heappush(pq, profit)
            total_profit += profit
            jobs_scheduled.append(i)
        elif pq and pq[0] < profit:  # Replace a lower profit job
            total_profit += profit - heapq.heappop(pq)
            heapq.heappush(pq, profit)


    return jobs_scheduled, total_profit 


class NodeTree:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other): 
        return self.freq < other.freq

def huffmanCoding(string):
    def generate_codes(node, prefix=""):
        if node is None:
            return {}
        if node.char:
            return {node.char: prefix}
        codes = {}
        codes.update(generate_codes(node.left, prefix + "0"))
        codes.update(generate_codes(node.right, prefix + "1"))
        return codes

    freq = {}
    for char in string:
        freq[char] = freq.get(char, 0) + 1


    heap = [NodeTree(char, f) for char, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = NodeTree(None, left.freq + right.freq)
        merged.left, merged.right = left, right
        heapq.heappush(heap, merged)

    huffman_tree = heapq.heappop(heap)
    return generate_codes(huffman_tree)

#print(huffmanCoding("hello world"))

#print(jobSequence([(100,2),(19,1),(27,2),(25,1),(15,1)]))
