# 二分法查找

执行示例：  
$ python binary_search/binary_search.py  
The list is: [1, 3, 4, 7, 8, 9, 12, 13, 16, 18, 19, 22, 24, 25], the target is 22  
The index of target 22 is: 11  
  
$ python binary_search/binary_search.py  
The list is: [1, 3, 4, 7, 8, 9, 12, 13, 16, 18, 19, 22, 24, 25], the target is 33  
Not found the target value in the list  
  
目标：  
给定一个有序列表list，在其中查找target值对应的位置索引。如果找不到target，返回-1。 
  
前置条件：  
二分法查找要求被查找的列表list是有序排列的。  
  
思路：  
要在给定的有序列表list中查找target，  
1、取得“区间”第一个元素的索引值low和最后一个元素的索引值high。  
2、计算“区间”中间那个元素的索引值mid，比较中间那个元素的值list[mid]和target：   
&emsp如果list[mid]等于target，说明找到了target，返回其索引；  
&emsp如果list[mid]大于target，说明target在上半区。low不变，high更新为mid-1，将“区间”更新为上半区；  
&emsp如果list[mid]小于target，说明target在下半区；low更新为mid+1，high不变，将“区间”更新为下半区。  
3、重复步骤1和2，直到找到target。如果low > high，说明已经遍历了整个列表，没有找到target，此时返回-1。  
  
```python
"""
Performing a binary search in a sorted list.

parameters:
list  - the sorted list
target - the target value to search for

return value：
The index of the found target value, or -1 if the target value is not found.
"""
def binary_search(list, target):
    low = 0
    high = len(list) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]

        if guess == target:
            return mid
        if guess > target:
            high = mid - 1
        else:
            low = mid + 1

    return -1
```
  
low和high分别代表当前查找区间的最低索引值和最高索引值，mid代表当前查找区间的中间位置索引值。  
mid的计算方法是 (low + high) // 2  
注意在Python中，  
"//"是整除运算符，结果会向下取整，例如 (0 + 5) // 2 = 2  
"/"是普通除法运算符，结果可能存在小数，例如 (0 + 5) / 2 = 2.5  
  
示例1：  
假设list是[0, 1, 2, 3, 4, 5]，target是5。  
第一轮查找，low等于0，high等于5，mid等于2。guess等于list[2]，即2。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 3，high不变。  
第二轮查找，low等于3，high等于5，mid等于4。guess等于list[4]，即4。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 5，high不变。  
第三轮查找，low等于5，high等于5，mid等于5。guess等于list[5]，即5。因为guess等于target，找到了target，返回其索引，即mid，即5。  
  
示例2：  
假设list是[0, 1, 2, 3, 4, 5]，target是0。  
第一轮查找，low等于0，high等于5，mid等于2。guess等于list[2]，即2。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 1，low不变。  
第二轮查找，low等于0，high等于1，mid等于0。guess等于list[0]，即0。因为guess等于target，找到了target，返回其索引，即mid，即0。  
  
示例3：  
假设list是[0, 1, 2, 3, 4, 5]，target是3。  
第一轮查找，low等于0，high等于5，mid等于2。guess等于list[2]，即2。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 3，high不变。  
第二轮查找，low等于3，high等于5，mid等于4。guess等于list[4]，即4。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 3，low不变。  
第三轮查找，low等于3，high等于3，mid等于3。guess等于list[3]，即3。因为guess等于target，找到了target，返回其索引，即mid，即3。  
  
示例4：  
假设list是[0, 1, 2, 3, 4, 5]，target是1。  
第一轮查找，low等于0，high等于5，mid等于2。guess等于list[2]，即2。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 1，low不变。  
第二轮查找，low等于0，high等于1，mid等于0。guess等于list[0]，即0。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 1，high不变。  
第三轮查找，low等于1，high等于1，mid等于1。guess等于list[1]，即1。因为guess等于target，找到了target，返回其索引，即mid，即1。  
  
示例5：  
假设list是[0, 1, 2, 3, 4, 5, 6]，target是5。  
第一轮查找，low等于0，high等于6，mid等于3。guess等于list[3]，即3。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 4，high不变。  
第二轮查找，low等于4，high等于6，mid等于5。guess等于list[5]，即5。因为guess等于target，找到了target，返回其索引，即mid，即5。  
  
示例6：  
假设list是[0, 1, 2, 3, 4, 5, 6]，target是2。  
第一轮查找，low等于0，high等于6，mid等于3。guess等于list[3]，即3。因为guess大于target，所以target在上半区，所以更新high = mid - 1 = 2，low不变。  
第二轮查找，low等于0，high等于2，mid等于1。guess等于list[1]，即1。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 2，high不变。  
第三轮查找，low等于2，high等于2，mid等于2。guess等于list[2]，即2。因为guess等于target，找到了target，返回其索引，即mid，即2。  
  
示例7：  
假设list是[0, 1, 2, 3, 4, 5, 6]，target是10。  
第一轮查找，low等于0，high等于6，mid等于3。guess等于list[3]，即3。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 4，high不变。  
第二轮查找，low等于4，high等于6，mid等于5。guess等于list[5]，即5。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 6，high不变。  
第三轮查找，low等于6，high等于6，mid等于6。guess等于list[6]，即6。因为guess小于target，所以target在下半区，所以更新low = mid + 1 = 7，high不变。  
第四轮查找，low等于7，high等于6，low大于了high，说明列表中没有target，返回-1。  
  
思考：  
到这里，我们可以理解一下为什么初始时设置low等于0，high等于len(list) - 1，而不是设置low等于1，high等于len(list)？  
因为这样设置low, high, mid都能代表列表值的索引值。  
因为这样low和high就正好是查找区间第一个元素和最后一个元素的索引，计算出来的mid值也正好是它对应值的索引。  
如果我们把low初始化为1，high初始化为len(list)，处理起来就不那么直观了。  
  
### License  
  
MIT
