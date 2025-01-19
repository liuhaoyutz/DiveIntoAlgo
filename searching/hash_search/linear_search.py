def linear_search(list, sum):
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if list[i] + list[j] == sum:
                return [i, j]
    return []

# examples：
nums = [2, 7, 11, 15]
sum = 9
print(f"The list is: {nums}")
print(f"The sum is {sum}")
print(linear_search(nums, sum))  # [0, 1]

nums = [11, 9, 33, 20, 15, 17, 16, 13]
sum = 48
print(f"The list is: {nums}")
print(f"The sum is {sum}")
print(linear_search(nums, sum))  # [2, 4