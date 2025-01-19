def merge_sort(list):
    if len(list) <= 1:
        return list

    # Breakdown: Find the midpoint to divide the array into two halves.
    mid = len(list) // 2
    left_half = list[:mid]
    right_half = list[mid:]

    # Recursively call merge_sort to sort the two halves.
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)

    # Merge the two sorted lists.
    return merge(left_sorted, right_sorted)

def merge(left, right):
    sorted_list = []
    i = j = 0

    # Merge two sorted lists.
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1

    # If there are any remaining elements, add them to the result list.
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])

    return sorted_list

# Example usage:
example_list = [64, 25, 12, 22, 11, 3, 32, 51, 42]
print("Unsorted list:", example_list)
example_list = merge_sort(example_list)
print("Sorted list:", example_list)