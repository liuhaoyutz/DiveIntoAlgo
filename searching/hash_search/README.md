# 哈希查找

执行示例：  
$ python hash_search.py  
The list is: [2, 7, 11, 15]  
The sum is 9  
[0, 1]  
The list is: [11, 9, 33, 20, 15, 17, 16, 13]  
The sum is 48  
[2, 4]  
  
目标：  
给定一个整数列表list和一个目标元素sum ，在列表中搜索“和”为sum的两个元素，并返回它们的索引。返回任意一个解即可。  
  
思路1，时间复杂度O(n^2)：  
2层for循环遍历list，检查哪2个元素满足条件，若找到，返回2个元素的索引。

```python
def linear_search(list, sum):
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if list[i] + list[j] == sum:
                return [i, j]
    return []
```

思路2，时间复杂度O(n)：  
用哈希表（在Python中是字典）来存储我们遍历过的元素及其索引。当我们遍历列表时，我们可以检查当前元素与sum的差值是否  
已经存在于哈希表中。如果存在，那么我们就找到了两个数，返回它们的索引。  
  
```python
def hash_search(list, sum):
    # Create an empty hash table to store the values and 
    # their corresponding indices.
    num_map = {}
    
    # Iterate through each element in the list.
    for index, num in enumerate(list):
        # Calculate the other number that needs to be found.
        complement = sum - num
        
        # Check if this number is already in the hash table.
        if complement in num_map:
            # If it exists, return the current index and 
            # the index of the corresponding value from the hash table.
            return [num_map[complement], index]
        
        # Add the current value and its index to the hash table.
        num_map[num] = index
    
    # If no solution is found, return an empty result.
    return []
```

### License  
  
MIT
