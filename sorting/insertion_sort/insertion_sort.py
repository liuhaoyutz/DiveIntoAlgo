def insertion_sort(list):
    # Starting from 1 because the first element is trivially sorted
    for i in range(1, len(list)):
        key = list[i]
        j = i - 1
        # Move elements of list[0..i-1], that are greater than key,
        # to one position ahead of their current position
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        # Place the key after the element just smaller than it.
        list[j + 1] = key

# Example usage:
list = [64, 25, 12, 22, 11, 3, 32, 51, 42]
print("Unsorted list:", list)
insertion_sort(list)
print("Sorted list:", list)