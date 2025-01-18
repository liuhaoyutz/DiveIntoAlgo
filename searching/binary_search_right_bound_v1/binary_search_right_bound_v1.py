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
    low, high = 0, len(list) - 1
    result = -1
    
    # Use binary search to find the insertion point.
    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]

        if guess == target:
            result = mid
            low = mid + 1
        elif guess < target:
            low = mid + 1
        else:
            high = mid - 1

    return result

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
