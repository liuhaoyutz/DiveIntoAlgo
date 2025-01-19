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

# examples：
sorted_list = [1, 3, 3, 3, 5, 5, 5, 5, 6]
print("The original list is:", sorted_list)

target = 5
index = binary_search_and_insert(sorted_list, target)
print(f"After insert {target}, the new list is: {sorted_list}, index: {index}")

target = 3
index = binary_search_and_insert(sorted_list, target)
print(f"After insert {target}, the new list is: {sorted_list}, index: {index}")

target = 2
index = binary_search_and_insert(sorted_list, target)
print(f"After insert {target}, the new list is: {sorted_list}, index: {index}")