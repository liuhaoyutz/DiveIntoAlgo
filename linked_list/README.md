# Python中的链表

Python标准库中并没有直接提供链表（linked list）这种数据结构。然而，Python提供了非常灵活和强大的列表（list）类型，  
它可以动态调整大小，并且在大多数情况下可以满足类似链表的需求。  
  
Python中的list和链表（linked list）是两种不同的数据结构，它们在内存布局、性能特征和适用场景上都有显著的区别。  
List:  
&emsp;Python的list是一个动态数组，在内存中以连续的块存储元素。  
&emsp;每个元素在内存中的位置是固定的，并且相邻元素在内存中也是相邻的。  
&emsp;这种布局使得随机访问非常快，因为你可以直接通过索引计算出元素的位置。  
Linked List:  
&emsp;链表由一系列节点组成，每个节点包含数据和指向下一个节点的引用（指针）。  
&emsp;节点在内存中可以是分散的，不一定是连续的。  
&emsp;每个节点只知道它的下一个节点在哪里，因此要访问某个节点，通常需要从头开始遍历链表。  
  
虽然Python没有内置的链表类型，但我们可以使用类来自定义实现。  
  
如果确实需要一个高效的链表实现，可以考虑使用一些第三方库，比如llist库，它提供了双向链表（dllist）和循环链表（cdllist）等功能。  
  
代码分析：    
linked_list.py文件实现了自定义链表。  
```python
class Node:
    def __init__(self, data=None):
        self.data = data  # Node's data
        self.next = None  # Reference to the next node
```
Node类代表链表的一个节点，包括2个成员，一个是节点的数据，另一个是指向下一个节点的指针。  
```python
class LinkedList:
    def __init__(self):
        self.head = None  # Initialize head as None
```
LinkedList类代表一个链表，head指向链表头。初始化链表时，head被设置为None。  
```python
    # Method to add a node at the end of the list
    def append(self, data):
        new_node = Node(data)
        if not self.head:  # If the list is empty, set the new node as the head
            self.head = new_node
        else:
            current = self.head
            while current.next:  # Traverse to the last node
                current = current.next
            current.next = new_node  # Append the new node
```
append函数用于将一个节点插入到链表尾部。  
如果head为None，即插入的是链表的第一个节点，则将head指向这个节点。  
如果head不为None，即插入的不是链表的第一个节点，则首先找到第一个节点，然后沿着next指针依次遍历链表，  
直到找到最后一个节点。将最后一个节点的next指针指向这个要插入的节点。这样，就把一个节点插入到了链表尾部。  
```python
    # Method to insert a node at a specific position
    def insert(self, data, position):
        new_node = Node(data)
        if position == 0:  # Insert at the beginning
            new_node.next = self.head
            self.head = new_node
        else:
            current = self.head
            index = 0
            while current and index < position - 1:  # Traverse to the position before the insertion point
                current = current.next
                index += 1
            if current:  # Insert after the found node
                new_node.next = current.next
                current.next = new_node
            else:
                print("Position out of range")
```
insert函数用于将一个节点插入到链表的position指定位置。  
如果position为0，则将节点插入到链表头部，让节点的next指针指向head（即原来的头节点），然后让head指向当前要插入的节点。  
否则，从头节点开始，沿着next指针遍历，直到找到索引为position-2的位置，把节点插入到索引为position-1的位置。  
```python
    # Method to delete a node by value
    def delete(self, key):
        current = self.head
        previous = None
        while current and current.data != key:
            previous = current
            current = current.next
        if current is None:  # Key not found
            print("Key not found")
        elif previous is None:  # Deleting the head node
            self.head = current.next
        else:
            previous.next = current.next  # Remove the node from the list
```
delete函数用于根据节点的值删除一个节点。  
从链表头开始沿着next指针遍历链表，直到找到值为key的节点。然后删除节点。  
```python
    # Method to search for a node by value
    def search(self, key):
        current = self.head
        index = 0
        while current and current.data != key:
            current = current.next
            index += 1
        if current:
            return index  # Return the index if the key is found
        else:
            return -1  # Return -1 if the key is not found
```
search函数用于在链表中查找值为key的节点。  
从链表头开始沿着next指针遍历链表，直到找到值为key的节点，返回其index。如果找不到，返回-1。  
```python
    # Method to traverse and print the list
    def traverse(self):
        current = self.head
        while current:
            print(current.data, end=" -> " if current.next else "\n")
            current = current.next
```
traverse函数用于遍历列表，依次打印链表每个节点的值。  
  
### License  
  
MIT
