# 二分法查找并插入
  
执行示例：  
$ python binary_search_and_insert.py  
The original array is: [1, 3, 5, 6]  
After insert 5, the new array is: [1, 3, 5, 5, 6], index: 2  
After insert 2, the new array is: [1, 2, 3, 5, 5, 6], index: 1  
After insert 7, the new array is: [1, 2, 3, 5, 5, 6, 7], index: 6  
After insert 0, the new array is: [0, 1, 2, 3, 5, 5, 6, 7], index: 0  
  
目标：  
给定一个有序数组array和一个元素target，数组不存在重复元素。  
将target插入数组array中，并保持其有序性。若数组中已存在元素target，则插入到其左边。  
返回插入后target在数组中的索引。  
  
前置条件：   
1、二分法查找要求被查找的数组array是有序排列的。  
2、数组array不存在重复元素。  
  
思路：  
要在给定的有序数组array中查找插入target的位置，  
1、取得“区间”第一个元素的索引值low和最后一个元素的索引值high。  
2、计算“区间”中间那个元素的索引值mid，比较中间那个元素的值array[mid]和target：  
   如果array[mid]等于target，说明找到了target，将target插入array[mid]左边，返回其索引值，即mid；  
   如果array[mid]大于target，说明target在上半区。low不变，high更新为mid-1，将“区间”更新为上半区；  
   如果array[mid]小于target，说明target在下半区；low更新为mid+1，high不变，将“区间”更新为下半区。  
3、重复步骤1和2，直到找到target。如果low > high，说明已经遍历了整个数组，没有找到target，此时的low值就是要插入位置的索引。  
  
```python
"""
Performing a binary search in a sorted array and insert the target value.

parameters:
array  - the sorted list
target - the target value to search for and insert

return value：
The index of the insert position.
"""
def binary_search_and_insert(array, target):
    low, high = 0, len(array) - 1
    
    # Use binary search to find the insertion point.
    while low <= high:
        mid = (low + high) // 2
        guess = array[mid]
        
        if guess == target:
            # If an equal element is found,
            # insert the new element to its left.
            array.insert(mid, target)
            return mid
        elif guess < target:
            low = mid + 1
        else:
            high = mid - 1
            
    # When the loop exits, low is the position to insert.
    array.insert(low, target)
    return low
```
  
low和high分别代表当前查找区间的最低索引值和最高索引值，mid代表当前查找区间的中间位置索引值。  
mid的计算方法是 (low + high) // 2  
注意在Python中，  
"//"是整除运算符，结果会向下取整，例如 (0 + 5) // 2 = 2   
"/"是普通除法运算符，结果可能存在小数，例如 (0 + 5) / 2 = 2.5  
  
示例1：  
假设array是[1, 3, 5, 6]，target是5。  
第一轮查找，low等于0，high等于3，mid等于1。guess等于array[1]，即3。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 2，high不变。  
第二轮查找，low等于2，high等于3，mid等于2。guess等于array[2]，即5。因为guess等于target，找到了与target相等的值，将target插入到其左边，返回插入点索引，即mid，即2。  
array变成[1, 3, 5, 5, 6]。  
  
示例2：  
假设array是[1, 3, 5, 5, 6]，target是2。  
第一轮查找，low等于0，high等于4，mid等于2。guess等于array[2]，即5。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 1，low不变。  
第二轮查找，low等于0，high等于1，mid等于0。guess等于array[0]，即1。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 1，high不变。  
第三轮查找，low等于1，high等于1，mid等于1。guess等于array[1]，即3。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 0，low不变。  
第四轮查找，low等于1，high等于0，退出循环。  
此时low，也就是1，就是插入点的索引。将target插入low指向的位置，返回low。  
array变成[1, 2, 3, 5, 5, 6]。  
  
示例3：  
假设array是[1, 2, 3, 5, 5, 6]，target是7。  
第一轮查找，low等于0，high等于5，mid等于2。guess等于array[2]，即3。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 3，high不变。  
第二轮查找，low等于3，high等于5，mid等于4。guess等于array[4]，即5。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 5，high不变。  
第三轮查找，low等于5，high等于5，mid等于5。guess等于array[5]，即6。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 6，high不变。  
第四轮查找，low等于6，hight等于5，退出循环。  
此时low，也就是6，就是插入点的索引。将target插入low指向的位置，返回low。  
array变成[1, 2, 3, 5, 5, 6, 7]。  
  
示例4：  
假设array是[1, 2, 3, 5, 5, 6, 7]，target是0。  
第一轮查找，low等于0，high等于6，mid等于3。guess等于array[3]，即5。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 2，low不变。  
第二轮查找，low等于0，high等于2，mid等于1。guess等于array[1]，即2。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 0，low不变。  
第三轮查找，low等于0，high等于0，mid等于0。guess等于array[0]，即1。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = -1，low不变。  
第四轮查找，low等于0，high等于-1，退出循环。  
此时low，也就是0，就是插入点的索引。将target插入low指向的位置，返回low。  
array变成[0, 1, 2, 3, 5, 5, 6, 7]。  
  
思考：  
到这里，我们可以理解一下为什么初始时设置low等于0，high等于len(array) - 1，而不是设置low等于1，high等于len(array)？  
因为这样设置low, high, mid都能代表数组值的索引值。  
因为这样low和high就正好是查找区间第一个元素和最后一个元素的索引，计算出来的mid值也正好是它对应值的索引。  
如果我们把low初始化为1，high初始化为len(array)，处理起来就不那么直观了。  
  
### License
  
MIT
