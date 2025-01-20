import heapq

# Create an empty heap (min-heap by default)
heap = []

# Insert elements into the heap
elements = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
for element in elements:
    heapq.heappush(heap, element)

print("The smallest element is:", heap[0])  # Get the minimum element

# Pop and return the smallest element from the heap
while heap:
    print("Popped:", heapq.heappop(heap))

# If you want to create a max-heap, you can invert the values when pushing and popping
max_heap = []
elements_max = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
for element in elements_max:
    # Push the negative of each element to simulate a max-heap
    heapq.heappush(max_heap, -element)

# Pop and return the largest element from the "max-heap"
while max_heap:
    print("Popped (max):", -heapq.heappop(max_heap))
