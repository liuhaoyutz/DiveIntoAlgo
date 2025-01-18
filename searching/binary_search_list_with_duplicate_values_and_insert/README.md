# 二分法查找并插入
  
执行示例：  
$ python binary_search_list_with_duplicate_values_and_insert.py  
The original list is: [1, 3, 3, 3, 5, 5, 5, 5, 6]  
After insert 5, the new list is: [1, 3, 3, 3, 5, 5, 5, 5, 5, 6], index: 4  
After insert 3, the new list is: [1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6], index: 1  
After insert 2, the new list is: [1, 2, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6], index: 1  
  
目标：  
给定一个有序列表list和一个元素target，列表可能存在重复元素。  
将target插入列表list中，并保持其有序性。若列表中已存在元素target，则插入到其最左边。  
返回插入后target在列表中的索引。  
  
前置条件：   
1、二分法查找要求被查找的列表list是有序排列的。  
2、列表list可能存在重复元素。  
  
思路：  
要在给定的有序列表list中查找插入target的位置，  
1、取得“区间”第一个元素的索引值low和最后一个元素的索引值high。  
2、计算“区间”中间那个元素的索引值mid，比较中间那个元素的值list[mid]和target：  
&emsp;如果list[mid]小于target，说明target在下半区；low更新为mid+1，high不变，将“区间”更新为下半区。  
&emsp;如果list[mid]大于target，说明target在上半区。low不变，high更新为mid-1，将“区间”更新为上半区；  
&emsp;如果list[mid]等于target，说明找到了target，但是需要继续向左探测，因为可能存在多个target，同样，low不变，high更新为mid-1，将”区间“更新为上半区；  
3、重复步骤1和2，直到low > high，说明已经遍历了整个列表，此时的low值就是要插入位置的索引。  
  
```python
"""
Performing a binary search in a sorted list and insert the target value.

parameters:
list  - the sorted list
target - the target value to search for and insert

return value：
The index of the insert position.
"""
def binary_search_and_insert(list, target):
    low, high = 0, len(list) - 1
    
    # Use binary search to find the insertion point.
    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]
        
        if guess < target:
            low = mid + 1
        else:
            high = mid - 1
            
    # When the loop exits, low is the position to insert.
    list.insert(low, target)
    return low
```
  
low和high分别代表当前查找区间的最低索引值和最高索引值，mid代表当前查找区间的中间位置索引值。  
mid的计算方法是 (low + high) // 2  
注意在Python中，  
"//"是整除运算符，结果会向下取整，例如 (0 + 5) // 2 = 2   
"/"是普通除法运算符，结果可能存在小数，例如 (0 + 5) / 2 = 2.5  
  
示例1：  
假设list是[1, 3, 3, 3, 5, 5, 5, 5, 6]，target是5。  
第一轮查找，low等于0，high等于8，mid等于4。guess等于list[4]，即5。因为guess等于target，所以更新high = mid - 1 = 3，low不变。  
第二轮查找，low等于0，high等于3，mid等于1。guess等于list[1]，即3。因为guess小于target，所以更新low = mid + 1 = 2，high不变。  
第三轮查找，low等于2，high等于3，mid等于2。guess等于list[2]，即3。因为guess小于target，所以更新low = mid + 1 = 3，high不变。  
第四轮查找，low等于3，high等于3，mid等于3。guess等于list[3]，即3。因为guess小于target，所以更新low = mid + 1 = 4，high不变。  
第五轮查找，low等于4，high等于3，退出循环。  
此时的low也就是4就是插入点的索引。将target插入到low指向的位置，返回low。  
list变成[1, 3, 3, 3, 5, 5, 5, 5, 5, 6]。  
  
示例2：  
假设list是[1, 3, 3, 3, 5, 5, 5, 5, 5, 6]，target是3。  
第一轮查找，low等于0，high等于9，mid等于4。guess等于list[4]，即5。因为guess大于target，所以更新high = mid - 1 = 3，low不变。  
第二轮查找，low等于0，high等于3，mid等于1。guess等于list[1]，即3。因为guess等于target，所以更新high = mid - 1 = 0，low不变。  
第三轮查找，low等于0，high等于0，mid等于0。guess等于list[0]，即1。因为guess小于target，所以更新low = mid + 1 = 1，high不变。  
第四轮查找，low等于1，high等于0，退出循环。  
此时low，也就是1，就是插入点的索引。将target插入low指向的位置，返回low。  
list变成list变成[1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6]。   
  
示例3：  
假设list是[1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6]，target是2。  
第一轮查找，low等于0，high等于10，mid等于5。guess等于list[5]，即5。因为guess大于target，所以更新high = mid - 1 = 4，low不变。   
第二轮查找，low等于0，high等于4，mid等于2。guess等于list[2]，即3。因为guess大于target，所以更新high = mid - 1 = 1，low不变。    
第三轮查找，low等于0，high等于1，mid等于0。guess等于list[0]，即1。因为guess小于target，所以更新low = mid + 1 = 1，high不变。   
第四轮查找，low等于1,high等于1，mid等于1。guess等于list[1]，即3。因为guess大于target，所以更新high = mid - 1 = 0，low不变。  
第五轮查找，low等于1，high等于0，退出循环。  
此时low，也就是1，就是插入点的索引。将target插入low指向的位置，返回low。  
list变成[1, 2, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6]。  
  
思考：  
到这里，我们可以理解一下为什么初始时设置low等于0，high等于len(list) - 1，而不是设置low等于1，high等于len(list)？  
因为这样设置low, high, mid都能代表列表值的索引值。  
因为这样low和high就正好是查找区间第一个元素和最后一个元素的索引，计算出来的mid值也正好是它对应值的索引。  
如果我们把low初始化为1，high初始化为len(list)，处理起来就不那么直观了。  
  
### License
  
MIT
