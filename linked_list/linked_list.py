class Node:
    def __init__(self, data=None):
        self.data = data  # Node's data
        self.next = None  # Reference to the next node

class LinkedList:
    def __init__(self):
        self.head = None  # Initialize head as None

    # Method to add a node at the end of the list
    def append(self, data):
        new_node = Node(data)
        if not self.head:  # If the list is empty, set the new node as the head
            self.head = new_node
        else:
            current = self.head
            while current.next:  # Traverse to the last node
                current = current.next
            current.next = new_node  # Append the new node

    # Method to insert a node at a specific position
    def insert(self, data, position):
        new_node = Node(data)
        if position == 0:  # Insert at the beginning
            new_node.next = self.head
            self.head = new_node
        else:
            current = self.head
            index = 0
            while current and index < position - 1:  # Traverse to the position before the insertion point
                current = current.next
                index += 1
            if current:  # Insert after the found node
                new_node.next = current.next
                current.next = new_node
            else:
                print("Position out of range")

    # Method to delete a node by value
    def delete(self, key):
        current = self.head
        previous = None
        while current and current.data != key:
            previous = current
            current = current.next
        if current is None:  # Key not found
            print("Key not found")
        elif previous is None:  # Deleting the head node
            self.head = current.next
        else:
            previous.next = current.next  # Remove the node from the list

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

    # Method to traverse and print the list
    def traverse(self):
        current = self.head
        while current:
            print(current.data, end=" -> " if current.next else "\n")
            current = current.next

if __name__ == "__main__":
    # Initialize linked list
    ll = LinkedList()

    # Append elements to the list
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print("After appending elements:")
    ll.traverse()

    # Insert an element at a specific position
    ll.insert(4, 1)  # Insert 4 at position 1
    print("After inserting 4 at position 1:")
    ll.traverse()

    # Delete an element by value
    ll.delete(2)
    print("After deleting element with value 2:")
    ll.traverse()

    # Search for an element by value
    search_key = 3
    index = ll.search(search_key)
    if index != -1:
        print(f"Found {search_key} at index {index}")
    else:
        print(f"{search_key} not found in the list")

    # Traverse the list
    print("Traversing the list:")
    ll.traverse()
