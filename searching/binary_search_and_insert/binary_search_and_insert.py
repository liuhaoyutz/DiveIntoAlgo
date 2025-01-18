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

# examples：
array = [1, 3, 5, 6]
print("The original array is:", array)

target = 5
index = binary_search_and_insert(array, target)
print(f"After insert {target}, the new array is: {array}, index: {index}")

target = 2
index = binary_search_and_insert(array, target)
print(f"After insert {target}, the new array is: {array}, index: {index}")

target = 7
index = binary_search_and_insert(array, target)
print(f"After insert {target}, the new array is: {array}, index: {index}")

target = 0
index = binary_search_and_insert(array, target)
print(f"After insert {target}, the new array is: {array}, index: {index}")