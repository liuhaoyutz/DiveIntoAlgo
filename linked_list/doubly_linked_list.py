class Node:
    def __init__(self, data=None):
        self.data = data  # Node's data
        self.prev = None  # Reference to the previous node
        self.next = None  # Reference to the next node

class DoublyLinkedList:
    def __init__(self):
        self.head = None  # Initialize head as None
        self.tail = None  # Initialize tail as None

    # Method to add a node at the end of the list
    def append(self, data):
        new_node = Node(data)
        if not self.head:  # If the list is empty
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    # Method to insert a node at the beginning of the list
    def prepend(self, data):
        new_node = Node(data)
        if not self.head:  # If the list is empty
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    # Method to insert a node after a specific node
    def insert_after(self, target_data, data):
        current = self.head
        while current and current.data != target_data:
            current = current.next
        if current is None:
            print("Target node not found.")
        else:
            new_node = Node(data)
            new_node.next = current.next
            new_node.prev = current
            if current.next:
                current.next.prev = new_node
            else:
                self.tail = new_node
            current.next = new_node

    # Method to delete a node by value
    def delete(self, key):
        current = self.head
        while current and current.data != key:
            current = current.next
        if current is None:  # Key not found
            print("Key not found")
        else:
            if current.prev:
                current.prev.next = current.next
            else:
                self.head = current.next  # Deleting the head node

            if current.next:
                current.next.prev = current.prev
            else:
                self.tail = current.prev  # Deleting the tail node

    # Method to search for a node by value
    def search(self, key):
        current = self.head
        index = 0
        while current and current.data != key:
            current = current.next
            index += 1
        if current:
            return index  # Return the index if the key is found
        else:
            return -1  # Return -1 if the key is not found

    # Method to traverse and print the list forward
    def traverse_forward(self):
        current = self.head
        while current:
            print(current.data, end=" <-> " if current.next else "\n")
            current = current.next

    # Method to traverse and print the list backward
    def traverse_backward(self):
        current = self.tail
        while current:
            print(current.data, end=" <-> " if current.prev else "\n")
            current = current.prev

if __name__ == "__main__":
    # Initialize doubly linked list
    dll = DoublyLinkedList()

    # Append elements to the list
    dll.append(1)
    dll.append(2)
    dll.append(3)
    print("After appending elements:")
    dll.traverse_forward()

    # Prepend an element
    dll.prepend(0)
    print("After prepending 0:")
    dll.traverse_forward()

    # Insert an element after a specific node
    dll.insert_after(2, 2.5)
    print("After inserting 2.5 after 2:")
    dll.traverse_forward()

    # Delete an element by value
    dll.delete(2)
    print("After deleting element with value 2:")
    dll.traverse_forward()

    # Search for an element by value
    search_key = 3
    index = dll.search(search_key)
    if index != -1:
        print(f"Found {search_key} at index {index}")
    else:
        print(f"{search_key} not found in the list")

    # Traverse the list forward and backward
    print("Traversing the list forward:")
    dll.traverse_forward()
    print("Traversing the list backward:")
    dll.traverse_backward()
