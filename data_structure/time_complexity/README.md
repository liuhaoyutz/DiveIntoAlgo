# 时间复杂度

时间复杂度是算法分析中的一个概念，用来描述算法执行所需的时间随着问题规模n增长而变化的趋势。   
  
n称为问题的规模。  
一个算法中的语句执行次数称为时间频度。记为T(n)。  
当n变化时，T(n)也会不断变化。我们想知道T(n)随着n的变化规律，所以引入了时间复杂度的概念。  
  
若存在某个函数f(n)，使得当n趋于无穷大时，T(n)/f(n)的极限值为不等于0的常数，则称f(n)是T(n)的同数量级函数。  
记作T(n) = O(f(n))，称O(f(n))为算法的时间复杂度。  
  
常见的时间复杂度  
时间复杂度按照从低到高排序：  
O(1) < O(logn) < O(n) < O(nlogn) < O(n^2) < O(2^n) < O(n!)  
  
1、常数阶O(1)  
```python
def constant_A(n: int):
    print(n)
```
这个函数无论n是多大，只是打印n的值，所以其时间复杂度是O(1)。  
  
```python
def constant_B(n: int) -> int:
    count = 0
    size = 100000
    for _ in range(size):
        count += 1
    return count
```
这个函数无论n是多大，都是循环100000次，即循环次数是固定的，与n无关，所以其时间复杂度还是常数阶O(1)。  
  
2、线性阶O(n)  
线性阶算法的执行时间相对于问题规模n以线性级别增长。线性阶通常出现在单层循环中。  
```python
def linear(n: int) -> int:
    count = 0
    for _ in range(n):
        count += 1
    return count
```
  
遍历数组和遍历链表等操作的时间复杂度均为O(n)，其中n为数组或链表的长度：  
```python
def array_traversal(nums: list[int]) -> int:
    count = 0
    for num in nums:
        count += 1
    return count
```
  
3、平方阶O(n^2)  
平方阶算法的执行时间相对于问题规模n以平方级别增长。平方阶通常出现在嵌套循环中，外层循环和内层循环的时间复杂度都为O(n)。  
```python
def quadratic(n: int) -> int:
    count = 0
    for i in range(n):
        for j in range(n):
            count += 1
    return count
```
  
4、指数阶O(2^n)  
下面的代码模拟细胞分裂的过程，时间复杂度为O(2^n) 。请注意，n表示分裂轮数，返回值count表示总分裂次数。  
```python
def exponential(n: int) -> int:
    count = 0
    base = 1
    for _ in range(n):
        for _ in range(base):
            count += 1
        base *= 2
    # count = 1 + 2 + 4 + 8 + .. + 2^(n-1) = 2^n - 1
    return count
```
  
5、对数阶O(logn)  
与指数阶相反，对数阶反映了“每轮缩减到一半”的情况。  
```python
def logarithmic(n: int) -> int:
    count = 0
    while n > 1:
        n = n / 2
        count += 1
    return count
```
  
6、线性对数阶O(nlogn)  
线性对数阶常出现于嵌套循环中，两层循环的时间复杂度分别为O(logn)和O(n)。  
```python
def linear_log_recur(n: int) -> int:
    if n <= 1:
        return 1
    # 一分为二，子问题的规模减小一半
    count = linear_log_recur(n // 2) + linear_log_recur(n // 2)
    # 当前子问题包含 n 个操作
    for _ in range(n):
        count += 1
    return count
```
  
7、阶乘阶O(n!)  
阶乘阶对应数学上的“全排列”问题。给定n个互不重复的元素，求其所有可能的排列方案，方案数量为：  
n! = n x (n - 1) x (n - 2) x ... x 2 x 1  
阶乘通常使用递归实现。  
```python
def factorial_recur(n: int) -> int:
    if n == 0:
        return 1
    count = 0
    # 从 1 个分裂出 n 个
    for _ in range(n):
        count += factorial_recur(n - 1)
    return count
```
  
### License  
  
MIT
