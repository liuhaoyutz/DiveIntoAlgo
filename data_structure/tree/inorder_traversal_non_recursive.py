class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal_non_recursive(root):
    stack = []
    current = root
    
    # Loop until there are no nodes left to process in the tree or stack
    while current is not None or stack:
        # Reach the leftmost node of the current node
        while current is not None:
            stack.append(current)  # Push the current node to the stack
            current = current.left  # Move to the left child
        
        # Backtrack from the empty subtree and visit the node at the top of the stack
        current = stack.pop()
        print(current.value)  # Process the node (here we just print the value)
        
        # We have visited the node and its left subtree. Now, it's time to visit the right subtree
        current = current.right

# Constructing a simple binary tree
#       1
#      / \
#     2   3
#    / \
#   4   5

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print("In-order traversal (non-recursive):")
inorder_traversal_non_recursive(root)
