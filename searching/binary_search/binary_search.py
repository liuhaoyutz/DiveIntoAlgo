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

sorted_list = [1, 3, 4, 7, 8, 9, 12, 13, 16, 18, 19, 22, 24, 25]
target_value = 22
print(f"The list is: {sorted_list}, the target is {target_value}")

result = binary_search(sorted_list, target_value)

if result != -1:
    print(f"The index of target {target_value} is: {result}")
else:
    print("Not found the target value in the list")