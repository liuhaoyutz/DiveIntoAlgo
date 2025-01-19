# 归并排序

执行示例：  
$ python merge_sort.py  
Unsorted list: [64, 25, 12, 22, 11, 3, 32, 51, 42]  
Sorted list: [3, 11, 12, 22, 25, 32, 42, 51, 64]  
  
目标：  
给定一个无序列表list，使用归并排序算法对其进行升序排序。   
  
思路：  
归并排序（Merge Sort）是一种基于分治法的高效、稳定的排序算法。它由约翰·冯·诺伊曼在1945年发明。  
归并排序的基本思想是将数组不断分割成更小的子数组，直到每个子数组只包含一个元素（因为单个元素的数组  
是天然有序的），然后逐步合并这些子数组，每次合并时都确保结果是有序的，最终得到完全排序的数组。  
  
1、分解：递归地将数组分成两半，直到每个子数组仅包含一个元素。  
2、合并：合并两个已经排序的子数组，生成一个新的有序数组。这个过程涉及到比较来自两个子数组的元素，并按顺序将它们放入新的数组中。  
3、递归终止条件：当子数组长度为1或0时，停止递归，因为这样的子数组已经是有序的。  
  
```python
def merge_sort(list):
    if len(list) <= 1:
        return list

    # Breakdown: Find the midpoint to divide the array into two halves.
    mid = len(list) // 2
    left_half = list[:mid]
    right_half = list[mid:]

    # Recursively call merge_sort to sort the two halves.
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)

    # Merge the two sorted lists.
    return merge(left_sorted, right_sorted)

def merge(left, right):
    sorted_list = []
    i = j = 0

    # Merge two sorted lists.
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1

    # If there are any remaining elements, add them to the result list.
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])

    return sorted_list
```

### License  
  
MIT
