class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def postorder_traversal_non_recursive(root):
    if root is None:
        return
    
    stack = []
    prev = None  # Keep track of the previously visited node
    current = root
    
    while stack or current is not None:
        # Reach the leftmost node of the current node
        while current is not None:
            stack.append(current)
            current = current.left
        
        # Backtrack from the empty subtree and visit the node at the top of the stack
        peek_node = stack[-1]
        
        # If there is no right child or the right child has been visited, process the node
        if peek_node.right is None or prev == peek_node.right:
            print(peek_node.value)  # Process the node (here we just print the value)
            prev = stack.pop()
        else:
            # Otherwise, move to the right child
            current = peek_node.right

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

print("Post-order traversal (non-recursive):")
postorder_traversal_non_recursive(root)
