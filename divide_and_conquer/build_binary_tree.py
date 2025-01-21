# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if not preorder or not inorder:
        return None

    # The first element of preorder is the root of the tree.
    root_val = preorder[0]
    root = TreeNode(root_val)

    # Find the index of the root in inorder traversal.
    root_index_in_inorder = inorder.index(root_val)

    # Divide inorder list into left and right subtrees.
    left_inorder = inorder[:root_index_in_inorder]
    right_inorder = inorder[root_index_in_inorder + 1:]

    # Divide preorder list into left and right subtrees.
    # Remove the root from the preorder list since it's already used.
    left_preorder = preorder[1:1 + len(left_inorder)]
    right_preorder = preorder[1 + len(left_inorder):]

    # Recursively build the left and right subtrees.
    root.left = buildTree(left_preorder, left_inorder)
    root.right = buildTree(right_preorder, right_inorder)

    return root

# Helper function to print the tree in pre-order for verification.
def print_preorder(node):
    if not node:
        return
    print(node.val, end=' ')
    print_preorder(node.left)
    print_preorder(node.right)

# Example usage:
if __name__ == "__main__":
    preorder_example = [3,9,20,15,7]
    inorder_example = [9,3,15,20,7]
    
    constructed_tree = buildTree(preorder_example, inorder_example)
    print("Preorder traversal of the constructed tree:")
    print_preorder(constructed_tree)
    print("\n")
