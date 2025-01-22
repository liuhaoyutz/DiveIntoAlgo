"""
Given an integer list list and a target element target, search for 
two elements in the list that add up to target, and return their indices.
Return any one of the solutions if multiple exist.

parameters:
list - the list to check
sum - the sum of two elements

return value：
The indices of the two elements
"""

def hash_search(list, sum):
    # Create an empty hash table to store the values and 
    # their corresponding indices.
    num_map = {}
    
    # Iterate through each element in the list.
    for index, num in enumerate(list):
        # Calculate the other number that needs to be found.
        complement = sum - num
        
        # Check if this number is already in the hash table.
        if complement in num_map:
            # If it exists, return the current index and 
            # the index of the corresponding value from the hash table.
            return [num_map[complement], index]
        
        # Add the current value and its index to the hash table.
        num_map[num] = index
    
    # If no solution is found, return an empty result.
    return []

# examples：
nums = [2, 7, 11, 15]
sum = 9
print(f"The list is: {nums}")
print(f"The sum is {sum}")
print(hash_search(nums, sum))  # [0, 1]

nums = [11, 9, 33, 20, 15, 17, 16, 13]
sum = 48
print(f"The list is: {nums}")
print(f"The sum is {sum}")
print(hash_search(nums, sum))  # [2, 4