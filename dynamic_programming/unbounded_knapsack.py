def unbounded_knapsack(weights, values, W):
    n = len(weights)
    # Initialize a DP table with dimensions (W+1)
    dp = [0] * (W + 1)
    
    # Fill the DP table
    for i in range(n):  # Iterate over each item
        for w in range(weights[i], W + 1):  # Update dp[w] for all capacities >= weights[i]
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[W]

# Example usage:
if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    W = 8
    print(f"The maximum value that can be put in a knapsack of capacity {W} is {unbounded_knapsack(weights, values, W)}")
