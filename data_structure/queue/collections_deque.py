from collections import deque

# Create an empty queue
queue = deque()

# Add elements to the queue using append() method (Enqueue)
queue.append('a')
queue.append('b')
queue.append('c')

print("Initial queue:", queue)

# Remove elements from the queue using popleft() method (Dequeue)
front_element = queue.popleft()
print("Dequeued element:", front_element)
print("Queue after dequeue:", queue)

# Peek at the front element of the queue without removing it
if queue:  # Check if the queue is not empty
    front_element = queue[0]
    print("Front element:", front_element)

# Get the size of the queue
queue_size = len(queue)
print("Queue size:", queue_size)

# Check if the queue is empty
is_empty = not queue
print("Is the queue empty?", is_empty)

# Continue to remove elements from the queue until it is empty
while queue:
    front_element = queue.popleft()
    print("Dequeued element:", front_element)

print("Is the queue empty now?", not queue)
