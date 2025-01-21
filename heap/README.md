# Python中的栈

堆（Heap）是一种特殊的完全二叉树，它可以分为两种类型：大顶堆（Max Heap）和小顶堆（Min Heap）。  
在大顶堆中，父节点的值总是大于或等于其子节点的值；而在小顶堆中，父节点的值总是小于或等于其子节点的值。  
堆通常用于实现优先队列，并且是堆排序算法的基础。  
  
堆的主要特性  
形状特性：堆是一棵完全二叉树，这意味着除了最后一层外，其他所有层都被完全填充，并且所有节点都尽可能靠左。  
堆序特性：对于大顶堆，任何给定节点的值总是大于或等于其子节点的值；对于小顶堆，任何给定节点的值总是小于或等于其子节点的值。  
  
Python标准库提供了heapq模块来实现小顶堆。  

代码分析：  
custom_heap.py文件实现了自定义小顶堆。  
```python
class MinHeap:
    def __init__(self):
        self.heap = []
```
初始化时，小顶堆是一个空列表。  
```python
    def parent(self, i):
        return (i - 1) // 2
```
parent函数返回索引为i的节点的父节点的索引。  
```python
    def insert_key(self, k):
        self.heap.append(k)
        i = len(self.heap) - 1
        self._heapify_up(i)
```
insert_key函数向堆中插入一个新的键值k。首先将键值添加到列表末尾，然后通过调用_heapify_up来维持小顶堆的性质。i是新插入键值的索引。  
```python
    def _heapify_up(self, i):
        while i != 0 and self.heap[self.parent(i)] > self.heap[i]:
            # Swap this node with its parent
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
```
_heapify_up是一个辅助函数，用于维护插入新节点后小顶堆的性质。它比较新节点与其父节点的值，并在必要时交换它们的位置，直到找到正确的位置。  
1、i是新插入节点的索引。parent(i)是新插入元素的父节点的索引。  
2、如果父节点的值大于新节点，交换他们的值。将i更新为parent(i)，重复步骤1，2。  
```python
    def extract_min(self):
        if len(self.heap) <= 0:
            return float("inf")
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root
```
extract_min函数移除并返回堆中的最小元素（即根节点）。如果堆中只有一个元素，则直接弹出并返回它。  
否则，用堆的最后一个元素替换根节点，然后调用_heapify_down函数从根开始向下调整堆以保持小顶堆的性质。  
```python
    def _heapify_down(self, i):
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self._heapify_down(smallest)
```
_heapify_down是一个辅助函数，用于从索引i开始向下调整堆。它确保子树满足小顶堆的性质，通过比较当前节点与其左右子节点，  
并在必要时与较小的那个子节点交换位置，递归地进行下去，直到不再违反小顶堆性质。  
当前节点的索引为i，则其左节点索引为2*i+1，其右节点索引为2*i+2。  
  
### License  
  
MIT
