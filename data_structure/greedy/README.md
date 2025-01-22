# 贪心算法

贪心算法（Greedy Algorithm）是一种在每个步骤中都做出局部最优选择的算法，期望通过这些局部最优解最终能够得到全局最优解。  
贪心算法并不总是能保证找到全局最优解，但在某些特定问题上，它可以有效地找到最优或接近最优的解决方案。  
  
贪心算法的特点  
局部最优选择：在每一步操作中，算法都会做出当前看起来最好的选择，而不考虑未来的影响。  
不可逆性：一旦做出了一个选择，就不会再改变这个选择。  
简单高效：由于不需要回溯和复杂的计算，贪心算法通常实现起来比较简单且运行速度快。  
  
活动选择问题  
问题描述：给定一组活动，每个活动都有开始时间和结束时间。选择尽可能多的互不冲突的活动进行参与。  
贪心策略：始终选择最早结束的活动，这样可以为后续活动留出更多的时间。  
activity_selection.py文件实现了活动选择问题。  
```python
def activity_selection(start, finish):
    # Sort activities based on their finish time
    activities = sorted(zip(finish, start))
    selected_activities = [activities[0]]
    
    for i in range(1, len(activities)):
        if activities[i][1] >= selected_activities[-1][0]:  # Start time is after the last finish time
            selected_activities.append(activities[i])
    
    return [(act[1], act[0]) for act in selected_activities]

# Example usage:
start_times = [1, 3, 0, 5, 8, 5]
finish_times = [2, 4, 6, 7, 9, 9]
selected = activity_selection(start_times, finish_times)
print("Selected activities:", selected)
```
1、使用zip(finish, start)将每个活动的结束时间和开始时间组合成一个元组列表。  
2、然后使用sorted()对这些元组按照第一个元素（即结束时间）进行排序。这样做的目的是为了确保我们总是优先选择最早结束的活动，  
&emsp;这符合贪心算法的思想——每次选择最早结束的活动可以为后续活动留出更多的时间。  
3、首先添加排序后的第一个活动，因为这是最早结束的活动。  
4、从第二个活动开始遍历所有排序后的活动。  
&emsp;对于每一个活动activities[i]，检查它的开始时间activities[i][1]是否大于或等于最后一个被选中活动的结束时间selected_activities[-1][0]。  
&emsp;如果条件满足，则说明当前活动与之前选择的所有活动都不冲突，因此可以将其加入到selected_activities中。  
6、最后，返回一个包含所有被选中活动的列表，其中每个活动的格式是 (start_time, finish_time)。  
  
区间覆盖问题  
问题描述：有一段线段 [a, b] 和若干条线段作为候选，要求使用最少数量的候选线段完全覆盖给定的线段 [a, b]。  
贪心策略：从左到右扫描，每次选择一个最靠左但又可以延伸到最右边的区间。  
interval_covering.py文件实现了区间覆盖问题。  
```python
def min_intervals_to_cover(intervals, a, b):
    intervals.sort(key=lambda x: x[1])  # Sort by end points
    
    cover = []
    current_end = a
    
    while current_end < b:
        farthest = -1
        next_interval = None
        
        for interval in intervals:
            if interval[0] <= current_end and interval[1] > farthest:
                farthest = interval[1]
                next_interval = interval
        
        if not next_interval:
            return None  # Cannot cover the entire segment
        
        cover.append(next_interval)
        current_end = farthest
    
    return cover

# Example usage:
intervals = [(1, 3), (2, 5), (3, 7), (6, 9)]
cover = min_intervals_to_cover(intervals, 1, 9)
if cover:
    print("Intervals used to cover:", cover)
else:
    print("Cannot cover the entire segment.")
```
1、使用sort()方法对intervals列表按照每个区间的结束点进行升序排序。  
&emsp;这一步是为了确保我们总是优先考虑那些可以延伸到最右边的区间，符合贪心算法的思想——每次选择能延伸最远的区间。  
2、外层while循环会一直执行，直到 current_end达到或超过目标线段的终点b，即整个线段都被覆盖。  
3、内层for循环寻找最佳区间。  
&emsp;farthest：记录当前找到的能够延伸到最右边位置的最大结束点。  
&emsp;next_interval：记录当前找到的最佳区间，即在不与已覆盖部分冲突的情况下延伸最远的那个区间。  
&emsp;遍历所有区间，检查是否满足两个条件：  
&emsp;区间的开始点interval[0]必须小于等于current_end，这意味着这个区间可以与已覆盖的部分相连。  
&emsp;区间的结束点interval[1]必须大于当前的farthest，这意味着它是目前找到的可以延伸最远的区间。  
&emsp;如果找到了符合条件的区间，则更新farthest和next_interval。  
4、如果遍历完所有区间后，next_interval仍然为 None，说明没有找到任何可以延伸覆盖范围的区间，因此无法完全覆盖目标线段。  
5、将找到的最佳区间添加到cover列表中。更新current_end为该区间的结束点farthest，表示我们现在覆盖到了新的最右端。  
6、当while循环结束时，意味着我们已经成功地用最少数量的区间覆盖了整个目标线段[a, b]，此时返回cover列表作为结果。  
  
分数背包问题  
问题描述：给定若干物品，每个物品有重量weight和价值value，并且有一个最大承重capacity的背包，目标是最大化装入背包的价值。  
与0-1背包不同的是，在分数背包问题中，我们可以取物品的一部分(最小1/weight)。  
贪心策略：按照单位重量的价值对物品排序，然后尽可能多地装入价值最高的物品，直到不能再装为止。  
fractional_knapsack.py文件实现了分数背包问题。  
```python
def fractional_knapsack(weights, values, capacity):
    # Create a list of tuples (value/weight, weight, value)
    items = [(v/w, w, v) for w, v in zip(weights, values)]
    items.sort(reverse=True, key=lambda item: item[0])  # Sort by value per unit weight
    
    total_value = 0
    remaining_capacity = capacity
    
    for value_per_weight, weight, value in items:
        if remaining_capacity == 0:
            break
        
        take_amount = min(weight, remaining_capacity)
        total_value += take_amount * value_per_weight
        remaining_capacity -= take_amount
    
    return total_value

# Example usage:
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
max_value = fractional_knapsack(weights, values, capacity)
print(f"Maximum value in knapsack: {max_value}")
```
1、使用列表推导式创建一个新的列表items，其中每个元素是一个元组(value_per_weight, weight, value)。value_per_weight 表示单位重量的价值，即 value / weight。  
2、使用sort()方法按照value_per_weight对items列表进行降序排序。即总是优先考虑单位重量价值最高的物品。  
3、用total_value记录当前装入背包的总价值。用remaining_capacity表示背包剩余的可用容量，初始值为 capacity。  
4、遍历排序后的 items 列表中的每个物品。  
&emsp;如果remaining_capacity已经为0，则直接跳出循环，因为没有更多空间可以装入物品。  
&emsp;否则，计算可以取走的物品数量take_amount，这是该物品的重量和背包剩余容量之间的最小值。  
&emsp;更新total_value，增加take_amount * value_per_weight，这代表了这部分物品的价值。  
&emsp;减少remaining_capacity，减去刚刚放入背包的物品重量take_amount。  
5、当遍历完所有物品或背包已满时，返回最终的 total_value，即能装入背包的最大总价值。  
  
### License  
  
MIT
