def knapsack(weights, values, W):
    n = len(weights)
    # Initialize a DP table with dimensions (n+1) x (W+1)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(W + 1):
            if w >= weights[i - 1]:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][W]

# Example usage:
if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    W = 5
    print(f"The maximum value that can be put in a knapsack of capacity {W} is {knapsack(weights, values, W)}")
