class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def preorder_traversal(node):
    if node is not None:
        print(node.value)  # Visit the root
        preorder_traversal(node.left)  # Traverse the left subtree
        preorder_traversal(node.right)  # Traverse the right subtree

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

print("Preorder traversal:")
preorder_traversal(root)
