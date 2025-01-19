def bubble_sort(list):
    n = len(list)
    # Traverse through all list elements
    for i in range(n):
        swapped = False
        # Last i elements are already in place, so the 
        # inner loop canavoid looking at the last i elements
        for j in range(0, n-i-1):
            # Swap if the element found is greater than the next element
            if list[j] > list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
                swapped = True
        # If no two elements were swapped by inner loop, then break as the list is sorted
        if not swapped:
            break

# Example usage:
list = [64, 25, 12, 22, 11, 3, 32, 51, 42]
print("Unsorted list:", list)
bubble_sort(list)
print("Sorted list:", list)