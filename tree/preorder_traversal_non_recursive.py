class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def preorder_traversal_non_recursive(root):
    if root is None:
        return
    
    # Initialize an empty stack and push the root node
    stack = [root]
    
    # Loop until the stack is empty
    while stack:
        # Pop a node from the stack and process it
        node = stack.pop()
        print(node.value)  # Process the node (here we just print the value)
        
        # Push right child first so that left child is processed first
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

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

print("Pre-order traversal (non-recursive):")
preorder_traversal_non_recursive(root)
