class Queue:
    def __init__(self):
        # Initialize an empty list to represent the queue
        self.items = []

    def is_empty(self):
        # Check if the queue is empty
        return len(self.items) == 0

    def enqueue(self, item):
        # Add a new item to the end of the queue
        self.items.append(item)

    def dequeue(self):
        # Remove and return the front item from the queue
        if not self.is_empty():
            # We use pop(0) to remove the first item of the list,
            # which corresponds to the front of the queue.
            return self.items.pop(0)
        else:
            raise IndexError("Dequeue from an empty queue")

    def peek(self):
        # Return the front item from the queue without removing it
        if not self.is_empty():
            return self.items[0]
        else:
            raise IndexError("Peek from an empty queue")

    def size(self):
        # Return the number of items in the queue
        return len(self.items)

# Example usage:
if __name__ == "__main__":
    # Create a new queue instance
    my_queue = Queue()

    # Enqueue elements into the queue
    my_queue.enqueue('a')
    my_queue.enqueue('b')
    my_queue.enqueue('c')

    print(f"Queue size: {my_queue.size()}")  # Should print the size of the queue

    print(f"Front element: {my_queue.peek()}")  # Should print the front element 'a'

    print(f"Dequeued element: {my_queue.dequeue()}")  # Should remove and print the front element 'a'
    print(f"New front element: {my_queue.peek()}")  # Should now print 'b'

    print(f"Is the queue empty? {my_queue.is_empty()}")  # Should print False

    while not my_queue.is_empty():
        print(f"Dequeue: {my_queue.dequeue()}")  # Dequeue all elements and print them

    print(f"Is the queue empty now? {my_queue.is_empty()}")  # Should print True
