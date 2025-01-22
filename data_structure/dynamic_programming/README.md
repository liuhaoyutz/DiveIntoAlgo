# 动态规划

动态规划（Dynamic Programming，简称DP）是一种在数学、管理科学、计算机科学、经济学和其他领域中用于求解复杂问题的算法设计技术。  
它通过将问题分解为更小的子问题，并保存这些子问题的解以避免重复计算，从而有效地解决问题。  
  
动态规划的基本思想  
最优子结构：如果一个问题的最优解可以由其子问题的最优解构造而来，那么这个问题就具有最优子结构性质。这意味着我们可以递归地定义问题的解。  
重叠子问题：在求解过程中，很多子问题会被多次计算。为了避免重复计算，动态规划会存储每个子问题的解，通常使用表格或数组来保存这些结果。这个过程被称为“记忆化”。  
  
动态规划的两种主要实现方式  
自底向上（Bottom-up）：这种方法从最小的子问题开始逐步构建到最终问题的解。通常使用迭代的方式填充一个表，确保在计算某个子问题之前，所有相关的更小子问题都已经被解决。  
自顶向下（Top-down）：也称为记忆化搜索，这种方法基于递归。当需要解决一个子问题时，首先检查是否已经计算过该子问题。如果是，则直接返回之前的结果；如果不是，则计算并保存结果。  
  
斐波那契数列问题  
每一项都是前两项的和。  
0,1,1,2,3,5,8,13,21,34,55,89,144,...  
  
直接使用递归的方法计算斐波那契数列第n项，效率很低，代码如下：  
```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
```
  
使用动态规划自底向上的思想，求解斐波那契数列的第n项，代码如下：  
```python
def fibonacci_bottom_up(n):
    """
    Bottom-up dynamic programming approach to compute the nth Fibonacci number.
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]
```
要求斐波那契数列第n项，一共有n+1个数，所以创建一个包含n+1个成员的列表dp。  
按自底向上的思想，先计算出dp[0], dp[1]，这样，计算dp[2]时，就可以直接利用之前已经解决的子问题，即dp[0]+dp[1]。  
依次类推，计算dp[n]时，可以利用之前已经解决的子问题，即dp[n] = dp[n-1]+dp[n-2]。  
  
使用动态规划自顶向下的思想，求解斐波那契数列的第n项，代码如下：  
```python
def fibonacci_top_down(n, memo=None):
    """
    Top-down dynamic programming approach with memoization to compute the nth Fibonacci number.
    """
    if memo is None:
        memo = {}
        
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_top_down(n - 1, memo) + fibonacci_top_down(n - 2, memo)
    return memo[n]
```
采用递归的方法，”递“的阶段，计算memo[n]时，需要计算memo[n-1]和memo[n-2], 依次类推，计算memo[2]时，需要计算memo[1]和memo[0]。  
关键在于，”归“的阶段，计算mem[3]时，需要用到mem[2]和mem[1]，此时的mem[2]和mem[1]都计算过并且保存下来了，可以直接拿过来用，不必再次计算了。这就是动态规划算法的好处。  
  
0-1背包问题（0-1 Knapsack Problem）：给定一组物品，每个物品都有自己的重量和价值，在限定的总重量内，如何选择装入背包中的物品以使得装入物品的总价值最大。  
  
问题定义  
输入：  
&emsp;一个正整数 W，表示背包的最大承重。  
&emsp;两个长度为 n 的数组 weights 和 values，分别表示 n 个物品的重量和对应的价值。  
输出：  
&emsp;返回可以装入背包的最大总价值。    
  
约束条件：  
&emsp;每个物品只能选择一次（即 "0-1" 的含义：要么不选这个物品，记为 0；要么选这个物品，记为 1），不能分割物品。  
&emsp;装入背包的物品总重量不能超过背包的最大承重 W。  
  
动态规划解法  
状态定义  
我们使用一个二维数组dp[i][w] 来表示从前i个物品中选择，并且背包当前剩余容量为w时所能获得的最大价值。  
  
状态转移方程  
对于每一个物品i和每一个可能的背包容量w：  
&emsp;如果不选择第i个物品，则 dp[i][w] = dp[i-1][w]；  
&emsp;如果选择第i个物品（前提是w >= weights[i]），则 dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i]] + values[i])。  
这里取两者中的较大值，确保在不超过背包容量的前提下获得最大价值。  
注意：  
这里就是动态规划的核心算法。计算dp[i][w]之前，已经计算过了其子问题dp[i-1][w]和dp[i-1][w-weights[i]]，  
无论是选择还是不选择第i个物品，计算dp[i][w]时，都可以利用之前已经计算过的子问题结果，避免重复计算。这就是动态规划的优势。  
  
边界条件  
&emsp;当没有物品可以选择时（i=0），无论背包容量多大，dp[0][w] = 0。  
&emsp;当背包容量为0时（w=0），无论有多少物品，dp[i][0] = 0。  
  
```python
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
```
  
### License  
  
MIT
