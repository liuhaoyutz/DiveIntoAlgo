"""
Performing a binary search in a sorted list for insert the target value.

parameters:
list  - the sorted list
target - the target value to search for insert

return value：
The index of the insert position.
"""
def binary_search_for_insert(list, target):
    low, high = 0, len(list) - 1
    
    # Use binary search to find the insertion point.
    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]
        
        if guess < target:
            low = mid + 1
        else:
            high = mid - 1

    return low

"""
Given a sorted list that may contain duplicate elements, 
return the index of the rightmost occurrence of element target
in the list. If target is not present in the list, return -1.

parameters:
list  - the sorted list
target - the target value to search

return value：
The index of the rightmost target position.
"""
def binary_search_right_bound(list, target):

    index_of_next = binary_search_for_insert(list, target + 1)
    # If target + 1 is not found, check whether the last element is target.
    if index_of_next == -1:
        if list and list[-1] == target:
            return len(list) - 1
        else:
            return -1
    
    # If target + 1 is found, then the rightmost target is the position right before it.
    if index_of_next > 0 and list[index_of_next - 1] == target:
        return index_of_next - 1
    else:
        return -1

# examples：
list = [1, 3, 3, 3, 5, 5, 5, 5, 6]
print("The original list is:", list)

target = 5
index = binary_search_right_bound(list, target)
print(f"The index of the leftmost {target} is {index}")

target = 3
index = binary_search_right_bound(list, target)
print(f"The index of the leftmost {target} is {index}")

target = 2
index = binary_search_right_bound(list, target)
print(f"The index of the leftmost {target} is {index}")
