class Node:
    def __init__(self, data=None):
        self.data = data  # Node's data
        self.next = None  # Reference to the next node

class CircularLinkedList:
    def __init__(self):
        self.head = None  # Initialize head as None

    # Method to append a node at the end of the list
    def append(self, data):
        if not self.head:  # If the list is empty
            self.head = Node(data)
            self.head.next = self.head  # Point to itself to form a circular link
        else:
            new_node = Node(data)
            current = self.head
            while current.next != self.head:  # Traverse to the last node
                current = current.next
            current.next = new_node  # Append the new node
            new_node.next = self.head  # Make it circular

    # Method to prepend a node at the beginning of the list
    def prepend(self, data):
        new_node = Node(data)
        if not self.head:  # If the list is empty
            self.head = new_node
            new_node.next = self.head  # Point to itself to form a circular link
        else:
            current = self.head
            while current.next != self.head:  # Traverse to the last node
                current = current.next
            current.next = new_node  # Make the last node point to the new node
            new_node.next = self.head  # Make the new node point to the old head
            self.head = new_node  # Update the head to the new node

    # Method to insert a node after a specific node
    def insert_after(self, target_data, data):
        if not self.head:
            print("The list is empty.")
            return
        new_node = Node(data)
        current = self.head
        while True:
            if current.data == target_data:
                break
            current = current.next
            if current == self.head:
                print("Target node not found.")
                return
        new_node.next = current.next
        current.next = new_node

    # Method to delete a node by value
    def delete(self, key):
        if not self.head:
            print("The list is empty.")
            return
        if self.head.next == self.head and self.head.data == key:  # Only one node in the list
            self.head = None
            return
        current = self.head
        prev = None
        while True:
            if current.data == key:
                if current == self.head:  # Deleting the head node
                    prev = self.head
                    while prev.next != self.head:
                        prev = prev.next
                    self.head = current.next
                    prev.next = self.head
                else:  # Deleting a non-head node
                    prev.next = current.next
                return
            prev = current
            current = current.next
            if current == self.head:
                print("Key not found")
                return

    # Method to search for a node by value
    def search(self, key):
        if not self.head:
            return -1  # Return -1 if the list is empty
        current = self.head
        index = 0
        while True:
            if current.data == key:
                return index  # Return the index if the key is found
            current = current.next
            index += 1
            if current == self.head:
                return -1  # Return -1 if the key is not found

    # Method to traverse and print the list
    def traverse(self):
        if not self.head:
            print("The list is empty.")
            return
        current = self.head
        while True:
            print(current.data, end=" -> " if current.next != self.head else "\n")
            current = current.next
            if current == self.head:
                break

if __name__ == "__main__":
    # Initialize circular linked list
    cll = CircularLinkedList()

    # Append elements to the list
    cll.append(1)
    cll.append(2)
    cll.append(3)
    print("After appending elements:")
    cll.traverse()

    # Prepend an element
    cll.prepend(0)
    print("After prepending 0:")
    cll.traverse()

    # Insert an element after a specific node
    cll.insert_after(2, 2.5)
    print("After inserting 2.5 after 2:")
    cll.traverse()

    # Delete an element by value
    cll.delete(2)
    print("After deleting element with value 2:")
    cll.traverse()

    # Search for an element by value
    search_key = 3
    index = cll.search(search_key)
    if index != -1:
        print(f"Found {search_key} at index {index}")
    else:
        print(f"{search_key} not found in the list")

    # Traverse the list
    print("Traversing the list:")
    cll.traverse()
