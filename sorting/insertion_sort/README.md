# 插入排序

执行示例：  
$ python insertion_sort.py  
Unsorted list: [64, 25, 12, 22, 11, 3, 32, 51, 42]  
Sorted list: [3, 11, 12, 22, 25, 32, 42, 51, 64]  
  
目标：  
给定一个无序列表list，使用选择插入算法对其进行升序排序。 
  
思路：  
1、List的第一个元素可以被认为已经排好序了。
2、取出下一个元素，暂存为key，然后在已经排序的元素序列中从后向前扫描。
3、如果该元素（已排序）大于key，将该元素移到下一位置。
4、重复步骤3，直到找到已排序的元素小于或者等于key的位置。
5、将key插入到该位置后面。
重复步骤2~5。
  
```python
def insertion_sort(list):
    # Starting from 1 because the first element is trivially sorted
    for i in range(1, len(list)):
        key = list[i]
        j = i - 1
        # Move elements of list[0..i-1], that are greater than key,
        # to one position ahead of their current position
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        # Place the key after the element just smaller than it.
        list[j + 1] = key
```

### License  
  
MIT
