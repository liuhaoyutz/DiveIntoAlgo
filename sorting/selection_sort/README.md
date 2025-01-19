# 选择排序

执行示例：  
$ python selection_sort.py  
Unsorted list: [64, 25, 12, 22, 11, 3, 32, 51, 42]  
Sorted list: [3, 11, 12, 22, 25, 32, 42, 51, 64]  
  
目标：  
给定一个无序列表list，使用选择排序算法对其进行排序。 
  
思路：  
将整个list看成前后2部分，前面部分是已经排好序的，后面部分是未排序的。  
第1轮循环，整个list都是未排序的，选择整个list最小的那个元素，放在list[0]位置。  
第2轮循环，list[0]是已经排序的，其他元素是未排序的。遍历未排序部分，选择最小的那个元素，放在list[1]位置。  
第3轮循环，list[0]和list[1]是已经排序的，其他元素是未排序的。遍历未排序部分，选择最小的那个元素，放在list[2]位置。  
第4轮循环，list[0], list[1], list[2]是已经排序的，其他元素是未排序的。遍历未排序部分，选择最小的那个元素，放在list[3]位置。  
... ...  
依次类推，直到遍历完整个list。  
  
```python
def selection_sort(list):
    # Traverse through all list elements
    for i in range(len(list)):
        # Assume the min is the first element
        min_idx = i
        # Test against elements after i to find the smallest
        for j in range(i+1, len(list)):
            # If this element is less, then it is the new minimum
            if list[j] < list[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element
        list[i], list[min_idx] = list[min_idx], list[i]
        # After each pass, the portion list[0:i+1] is sorted
```

### License  
  
MIT
