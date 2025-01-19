def quick_sort(list):
    if len(list) <= 1:
        return list
    else:
        pivot = list[len(list) // 2]
        left = [x for x in list if x < pivot]
        middle = [x for x in list if x == pivot]
        right = [x for x in list if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

# Example usage:
example_list = [64, 25, 12, 22, 11, 3, 32, 51, 42]
print("Unsorted list:", example_list)
example_list = quick_sort(example_list)
print("Sorted list:", example_list)