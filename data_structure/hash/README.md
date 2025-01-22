# Python中的哈希表

哈希表（Hash Table），也称为散列表，是一种数据结构，它实现了关联数组抽象数据类型，即一个可以将键（key）映射到值（value）的数据结构。  
哈希表使用一种叫做哈希函数的算法来计算一个给定键的索引，这个索引决定了该键对应的值在底层数组中的存储位置。  
  
哈希表的关键特性是提供平均情况下的常数时间复杂度O(1)的查找、插入和删除操作，这使得它们非常高效，尤其是在处理大量数据时。  
然而，在最坏的情况下（例如所有键都映射到同一个索引，导致大量的碰撞），这些操作的时间复杂度可能会退化到O(n)，其中n是哈希表中元素的数量。  
  
当两个不同的键通过哈希函数计算得到相同的索引时，会发生哈希冲突。为了处理这种情况，哈希表通常会采用以下几种策略之一：  
  
链地址法（Separate Chaining）：每个桶包含一个链表，所有哈希到同一位置的键都会被添加到这个链表中。  
开放寻址（Open Addressing）：当发生冲突时，在表中寻找下一个空闲的位置来存储新元素。  
  
在Python中，字典（dict）就是一种哈希表的实现。  
  
代码分析：  
custom_hash.py文件实现了自定义哈希表。  
```python
class HashTable:
    def __init__(self, size=10):
        # Initialize the hash table with empty bucket list entries.
        self.size = size
        self.table = [[] for _ in range(self.size)]
```
类HashTable代表一个哈希表。size是哈希表的大小。table是一个bucket的列表。bucket也是一个列表，其每个成员将是(key, value)元组。  
```python
    def _hash(self, key):
        # A simple hash function that uses Python's built-in hash() function
        return hash(key) % self.size
```
_hash函数用于生成哈希值。调用了Python内置的hash()函数。  
```python
    def put(self, key, value):
        # Insert or update the value associated with the key
        hash_index = self._hash(key)
        bucket = self.table[hash_index]
        
        found = False
        for index, (k, v) in enumerate(bucket):
            if k == key:
                bucket[index] = (key, value)  # Update existing key-value pair
                found = True
                break
        
        if not found:
            bucket.append((key, value))  # Add new key-value pair
```
put函数用于将(key, value)元组添加到对应的bucket中，如果key已经存在，则更新value。  
key通过调用_hash函数生成哈希表index，找到对应的bucket。  
bucket也是一个列表，遍历bucket，如果已经存在key，则更新value。如果bucket中不存在key，则将(key, value)元组追加到bucket列表的尾部。  
```python
    def get(self, key):
        # Retrieve the value associated with the key
        hash_index = self._hash(key)
        bucket = self.table[hash_index]
        
        for k, v in bucket:
            if k == key:
                return v  # Return the value if key is found
        
        raise KeyError(f"Key '{key}' not found")
```
get函数用于取得key对应的value。  
key通过调用_hash函数生成哈希表index，找到对应的bucket。  
bucket也是一个列表，遍历bucket，查找有没有key，如果找到，返回对应的value。  
```python
    def remove(self, key):
        # Remove the key-value pair from the hash table
        hash_index = self._hash(key)
        bucket = self.table[hash_index]
        
        for index, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[index]  # Remove the key-value pair
                return
        
        raise KeyError(f"Key '{key}' not found")
```
remove函数用于从哈希表中删除(key, value)。  
Key通过调用_hash函数生成哈希表index，找到对应的bucket。  
遍历bucket，查找key，如果找到，删除(key, value)。  
```python
    def __str__(self):
        # String representation of the hash table
        pairs = []
        for bucket in self.table:
            for key, value in bucket:
                pairs.append(f"{key}: {value}")
        return "{" + ", ".join(pairs) + "}"
```
__str__函数用于显示整个哈希表内容。  
  
### License  
  
MIT
