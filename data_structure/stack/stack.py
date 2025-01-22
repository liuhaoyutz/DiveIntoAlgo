class Stack:
    def __init__(self):
        # Initialize an empty list as the internal representation of the stack
        self.items = []

    def is_empty(self):
        # Check if the stack is empty
        return len(self.items) == 0

    def push(self, item):
        # Add an item to the top of the stack
        self.items.append(item)

    def pop(self):
        # Remove and return the top item of the stack
        # If the stack is empty, raise an exception
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("Pop from an empty stack")

    def peek(self):
        # Return the top item of the stack without removing it
        # If the stack is empty, raise an exception
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError("Peek from an empty stack")

    def size(self):
        # Return the number of items in the stack
        return len(self.items)

# Example usage:
if __name__ == "__main__":
    # Create a new stack instance
    my_stack = Stack()

    # Push elements onto the stack
    my_stack.push('a')
    my_stack.push('b')
    my_stack.push('c')

    print(f"Stack size: {my_stack.size()}")  # Should print the size of the stack

    print(f"Top element: {my_stack.peek()}")  # Should print the top element 'c'

    print(f"Popped element: {my_stack.pop()}")  # Should remove and print the top element 'c'
    print(f"New top element: {my_stack.peek()}")  # Should now print 'b'

    print(f"Is the stack empty? {my_stack.is_empty()}")  # Should print False

    while not my_stack.is_empty():
        print(f"Popping: {my_stack.pop()}")  # Pop all elements and print them

    print(f"Is the stack empty now? {my_stack.is_empty()}")  # Should print True
