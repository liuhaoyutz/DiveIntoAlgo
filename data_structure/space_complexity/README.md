# 空间复杂度

空间复杂度是算法分析中的一个概念，用来描述算法执行所需的内存空间随着问题规模n增长而变化的趋势。   
  
常见的空间复杂度  
空间复杂度按照从低到高排序：  
O(1) < O(logn) < O(n) < O(n^2) < O(2^n)  
  
1、常数阶O(1)  
常数阶常见于数量与输入数据大小n无关的常量、变量、对象。  
需要注意的是，在循环中初始化变量或调用函数而占用的内存，在进入下一循环后就会被释放，  
因此不会累积占用空间，空间复杂度仍为O(1)。  
```python
def function() -> int:
    """函数"""
    # 执行某些操作
    return 0

def constant(n: int):
    # 常量、变量、对象占用 O(1) 空间
    a = 0
    nums = [0] * 10000
    node = ListNode(0)
    # 循环中的变量占用 O(1) 空间
    for _ in range(n):
        c = 0
    # 循环中的函数占用 O(1) 空间
    for _ in range(n):
        function()
```
  
2、线性阶O(n)  
线性阶常见于元素数量与n成正比的数组、链表、栈、队列等：  
```python
def linear(n: int):
    # 长度为 n 的列表占用 O(n) 空间
    nums = [0] * n
    # 长度为 n 的哈希表占用 O(n) 空间
    hmap = dict[int, str]()
    for i in range(n):
        hmap[i] = str(i)
```
  
下面的函数的递归深度为n，即同时存在n个未返回的linear_recur()函数，使用O(n)大小的栈帧空间：   
```python
def linear_recur(n: int):
    """线性阶（递归实现）"""
    print("递归 n =", n)
    if n == 1:
        return
    linear_recur(n - 1)
```
  
3、平方阶O(n^2)  
平方阶常见于矩阵和图，元素数量与n成平方关系：   
```python
def quadratic(n: int):
    # 二维列表占用 O(n^2)空间
    num_matrix = [[0] * n for _ in range(n)]
```
  
下面的函数的递归深度为n，在每个递归函数中都初始化了一个数组，长度分别为n, n-1, n-2, ..., 2, 1, 平均长度为n/2，因此总体占用O(n^2)空间。  
```python
def quadratic_recur(n: int) -> int:
    if n <= 0:
        return 0
    # 数组 nums 长度为 n, n-1, ..., 2, 1
    nums = [0] * n
    return quadratic_recur(n - 1)
```
  
4、指数阶O(2^n)  
指数阶常见于二叉树。层数为n的“满二叉树”的节点数量为2^n - 1，占用O(2^n)空间。   
```python
def build_tree(n: int) -> TreeNode | None:
    """指数阶（建立满二叉树）"""
    if n == 0:
        return None
    root = TreeNode(0)
    root.left = build_tree(n - 1)
    root.right = build_tree(n - 1)
    return root
```
  
5、对数阶O(logn)  
对数阶常见于分治算法。例如归并排序，输入长度为n的数组，每轮递归将数组从中点处划分为两半，形成高度为logn的递归树，使用O(logn)栈帧空间。  

### License  
  
MIT
