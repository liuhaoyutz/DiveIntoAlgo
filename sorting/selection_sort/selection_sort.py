def selection_sort(list):
    # Traverse through all list elements
    for i in range(len(list)):
        # Assume the min is the first element
        min_idx = i
        # Test against elements after i to find the smallest
        for j in range(i+1, len(list)):
            # If this element is less, then it is the new minimum
            if list[j] < list[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element
        list[i], list[min_idx] = list[min_idx], list[i]
        # After each pass, the portion list[0:i+1] is sorted

# Example usage:
example_list = [64, 25, 12, 22, 11, 3, 32, 51, 42]
print("Unsorted list:", example_list)
selection_sort(example_list)
print("Sorted list:", example_list)