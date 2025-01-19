# 二分法查找右边界_v2
  
执行示例：  
$ python binary_search_right_bound_v2.py  
The original list is: [1, 3, 3, 3, 5, 5, 5, 5, 6]  
The index of the leftmost 5 is 7  
The index of the leftmost 3 is 3  
The index of the leftmost 2 is -1  
    
目标：  
给定一个有序列表list和一个元素target，列表可能存在重复元素。  
返回list中最右一个target的索引。若list中不包含该元素，则返回-1。  
  
前置条件：   
1、二分法查找要求被查找的列表list是有序排列的。  
2、列表list可能存在重复元素。  
  
思路：  
将查找最右一个target转化为查找最左一个target + 1。 
  
```python
"""
Performing a binary search in a sorted list for insert the target value.

parameters:
list  - the sorted list
target - the target value to search for insert

return value：
The index of the insert position.
"""
def binary_search_for_insert(list, target):
    low, high = 0, len(list) - 1
    
    # Use binary search to find the insertion point.
    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]
        
        if guess < target:
            low = mid + 1
        else:
            high = mid - 1

    return low

"""
Given a sorted list that may contain duplicate elements, 
return the index of the rightmost occurrence of element target
in the list. If target is not present in the list, return -1.

parameters:
list  - the sorted list
target - the target value to search

return value：
The index of the rightmost target position.
"""
def binary_search_right_bound(list, target):

    index_of_next = binary_search_for_insert(list, target + 1)
    # If target + 1 is not found, check whether the last element is target.
    if index_of_next == -1:
        if list and list[-1] == target:
            return len(list) - 1
        else:
            return -1
    
    # If target + 1 is found, then the rightmost target is the position right before it.
    if index_of_next > 0 and list[index_of_next - 1] == target:
        return index_of_next - 1
    else:
        return -1
```
  
low和high分别代表当前查找区间的最低索引值和最高索引值，mid代表当前查找区间的中间位置索引值。  
mid的计算方法是 (low + high) // 2  
注意在Python中，  
"//"是整除运算符，结果会向下取整，例如 (0 + 5) // 2 = 2   
"/"是普通除法运算符，结果可能存在小数，例如 (0 + 5) / 2 = 2.5  
  
### License
  
MIT
