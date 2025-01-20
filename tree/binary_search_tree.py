class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, root, key):
        # If the tree is empty, return a new node
        if root is None:
            return TreeNode(key)
        
        # Otherwise, recur down the tree
        if key < root.val:
            root.left = self.insert(root.left, key)
        elif key > root.val:
            root.right = self.insert(root.right, key)

        # Return the (unchanged) node pointer
        return root

    def search(self, root, key):
        # Base cases: root is null or key is present at root
        if root is None or root.val == key:
            return root
        
        # Key is greater than root's key
        if root.val < key:
            return self.search(root.right, key)
        
        # Key is smaller than root's key
        return self.search(root.left, key)

    def inorder_traversal(self, root):
        if root:
            # First recur on left child
            self.inorder_traversal(root.left)
            # Then print the data of node
            print(root.val, end=' ')
            # Now recur on right child
            self.inorder_traversal(root.right)

# Example usage:
if __name__ == "__main__":
    bst = BinarySearchTree()
    
    # Insert nodes into the binary search tree
    keys = [50, 30, 20, 40, 70, 60, 80]
    for key in keys:
        bst.root = bst.insert(bst.root, key)

    print("In-order traversal of the constructed BST:")
    bst.inorder_traversal(bst.root)  # Expected output: 20 30 40 50 60 70 80
    
    # Searching for a value in the BST
    search_key = 40
    result = bst.search(bst.root, search_key)
    if result:
        print(f"\nKey {search_key} found in BST.")
    else:
        print(f"\nKey {search_key} not found in BST.")
