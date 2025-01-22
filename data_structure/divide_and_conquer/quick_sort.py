def quick_sort(arr):
    # If the array length is less than or equal to 1, it is already sorted, return it directly.
    if len(arr) <= 1:
        return arr
    
    # Choose a pivot element. Here we choose the first element of the list.
    pivot = arr[0]
    
    # Create two lists: one for elements less than or equal to the pivot,
    # and another for elements greater than the pivot.
    less_than_pivot = [x for x in arr[1:] if x <= pivot]
    greater_than_pivot = [x for x in arr[1:] if x > pivot]
    
    # Recursively call quick_sort on both sublists and combine the results.
    # The less_than_pivot sublist comes before the pivot, which comes before the greater_than_pivot sublist.
    return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)

# Example usage
if __name__ == "__main__":
    unsorted_array = [3, 6, 8, 10, 1, 2, 1]
    print("Unsorted array:", unsorted_array)
    sorted_array = quick_sort(unsorted_array)
    print("Sorted array:", sorted_array)
