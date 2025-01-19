# 冒泡排序

执行示例：  
$ python bubble_sort.py  
Unsorted list: [64, 25, 12, 22, 11, 3, 32, 51, 42]  
Sorted list: [3, 11, 12, 22, 25, 32, 42, 51, 64]  
  
目标：  
给定一个无序列表list，使用冒泡排序算法对其进行排序。 
  
思路：  
将整个list看成前后2部分，前面部分是未排序的，后面部分是排好序的。  
第1轮循环，整个list都是未排序的，选择整个list最大的那个元素，放在list[n-1]位置，即list最后一个位置。  
第2轮循环，list[n-1]是已经排序的，其他元素是未排序的。遍历未排序部分，选择最大的那个元素，放在list[n-2]位置。  
第3轮循环，list[n-2]和list[n-1]是已经排序的，其他元素是未排序的。遍历未排序部分，选择最大的那个元素，放在list[n-3]位置。  
第4轮循环，list[n-3], list[n-2], list[n-1]是已经排序的，其他元素是未排序的。遍历未排序部分，选择最大的那个元素，放在list[n-4]位置。  
... ...  
依次类推，直到遍历完整个list。  
  
```python
def bubble_sort(list):
    n = len(list)
    # Traverse through all list elements
    for i in range(n):
        swapped = False
        # Last i elements are already in place, so the 
        # inner loop canavoid looking at the last i elements
        for j in range(0, n-i-1):
            # Swap if the element found is greater than the next element
            if list[j] > list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
                swapped = True
        # If no two elements were swapped by inner loop, then break as the list is sorted
        if not swapped:
            break
```

### License  
  
MIT
