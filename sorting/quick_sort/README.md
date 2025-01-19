# 快速排序

执行示例：  
$ python quick_sort.py  
Unsorted list: [64, 25, 12, 22, 11, 3, 32, 51, 42]  
Sorted list: [3, 11, 12, 22, 25, 32, 42, 51, 64]   
  
目标：  
给定一个无序列表list，使用快速排序算法对其进行升序排序。   
  
思路：  
快速排序（Quicksort）是一种高效的、基于分治法的排序算法。它由C. A. R. Hoare在1960年提出。  
快速排序的基本思想是通过选择一个“基准”元素（pivot），将数组分为两部分：一部分的所有元素都比基准小，  
另一部分的所有元素都比基准大，然后递归地对这两部分进行同样的操作。  
  
1、选择基准：从数组中选择一个元素作为基准。这个选择可以有很多策略，比如总是选第一个元素、最后一个元素、中间元素或者随机选择一个元素。  
2、分区操作：重新排列数组（或列表），使得所有比基准小的元素都排在基准前面，所有比基准大的元素都排在基准后面，这个过程称为分区（partitioning）。  
3、递归排序：分别对基准两边的子数组递归地应用上述两个步骤。递归的终止条件是子数组的大小为零或一，此时该子数组已经是有序的。  
  
注意：下面这个实现不是原地排序的，因为它创建了新的列表来保存left、middle和right。对于大型数据集，更有效的实现方式是使用原地排序。  
```python
def quick_sort(list):
    if len(list) <= 1:
        return list
    else:
        pivot = list[len(list) // 2]
        left = [x for x in list if x < pivot]
        middle = [x for x in list if x == pivot]
        right = [x for x in list if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
```

### License  
  
MIT
