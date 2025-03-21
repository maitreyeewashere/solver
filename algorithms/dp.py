def knapsack01(cap, items):
    #items[i] = (profit, weight)
    cap = int(cap)
    items = [(int(p),int(w)) for p,w in items]
    n = len(items)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(cap + 1):
            if items[i - 1][1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - items[i - 1][1]] + items[i - 1][0])
            else:
                dp[i][w] = dp[i - 1][w]

    return {"profit": dp[n][cap], "dp_table": dp}

def lcs(str1, str2):
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return {"length": dp[n][m], "dp_table": dp}


def mcm(arr):
    n = len(arr)
    dp = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 0

    for l in range(2, n):
        for i in range(1, n - l + 1):
            j = i + l - 1
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + arr[i - 1] * arr[k] * arr[j]
                dp[i][j] = min(dp[i][j], cost)

    dp_serializable = [[x if x != float('inf') else 10**9 for x in row] for row in dp]

    return {"cost": dp[1][n - 1], "dp_table": dp_serializable}

    
