# Python中的栈

栈（Stack）是一种后进先出（LIFO, Last In First Out）的数据结构。这意味着最后被添加到栈中的元素将是第一个被移除的元素。  
栈的概念类似于现实生活中的一叠盘子：只能从顶部拿取或添加一个盘子，底部的盘子只有在其上面的所有盘子都被移走之后才能被触及。  
  
在计算机科学和编程语言中，栈通常用于实现函数调用、表达式求值与解析、内存管理等场景。  
Python本身没有内置的栈类型，但是可以通过Python的列表（list）来非常方便地模拟栈的行为，因为Python的列表已经提供了所有必要的操作，  
如append()（用来添加元素，相当于栈的push操作）和pop()（用来移除并返回最后一个元素，相当于栈的pop操作）。  
  
代码分析：  
stack.py文件通过list实现了自定义stack。  
```python
class Stack:
    def __init__(self):
        # Initialize an empty list as the internal representation of the stack
        self.items = []
```
Stack类初始时是一个空列表。  
```python
    def is_empty(self):
        # Check if the stack is empty
        return len(self.items) == 0
```
is_empty函数用于判断Stack是否为空。即判断列表items是否为空。  
```python
    def push(self, item):
        # Add an item to the top of the stack
        self.items.append(item)
```
push函数用于将item压入Stack，即放在列表items的尾部。  
```python
    def pop(self):
        # Remove and return the top item of the stack
        # If the stack is empty, raise an exception
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("Pop from an empty stack")
```
poop函数用于将Stack顶部的元素出栈。即将列表items的尾节点弹出。  
```python
    def peek(self):
        # Return the top item of the stack without removing it
        # If the stack is empty, raise an exception
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError("Peek from an empty stack")
```
peek函数用于取得栈顶元素的值。即取得列表items的尾节点的值。  
```python
    def size(self):
        # Return the number of items in the stack
        return len(self.items)
```
size函数返回栈大小。即返回列表items的大小。  
  
### License  
  
MIT
