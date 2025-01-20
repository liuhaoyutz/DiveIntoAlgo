# Python中的队列

在Python中，队列（Queue）是一种先进先出（FIFO, First In First Out）的数据结构。这意味着最早被添加到队列中的元素将是第一个被移除的元素。  
队列的概念类似于排队等候的队伍：首先到达的人会最先离开。  
  
Python标准库提供了queue模块，其中包含了多种队列实现，例如Queue、LifoQueue（栈）、PriorityQueue等。  
对于简单的队列操作，也可以直接使用列表，但要注意，列表的插入和删除操作在列表头部时效率较低（O(n)），因为所有其他元素都需要移动。  
而collections.deque则专门为快速从两端进行插入和删除操作设计，因此是更好的选择。  
  
如果需要一个线程安全的队列，可以使用queue.Queue类，它特别适用于多线程编程。  
  
### License  
  
MIT
