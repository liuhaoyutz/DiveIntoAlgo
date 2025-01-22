# Python中的二叉树

二叉树（Binary Tree）是一种每个节点最多有两个子节点的树形数据结构。  
  
二叉树的特点  
&emsp;每个节点最多有两棵子树，分别称为左子树和右子树。  
&emsp;左子树和右子树是有顺序的，次序不能任意颠倒。  
&emsp;即使树中某节点只有一棵子树，也要区分它是左子树还是右子树。  
  
二叉树的种类  
&emsp;满二叉树（Full Binary Tree）：除了叶子节点外，每一个节点都恰好有两个子节点。  
&emsp;完全二叉树（Complete Binary Tree）：除最后一层外，其它各层的节点数目均已达最大值，且最后一层的节点都集中在该层最左边。  
&emsp;平衡二叉树（Balanced Binary Tree）：又被称为AVL树，它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。  
    
二叉搜索树（Binary Search Tree, BST）是一种特殊的二叉树，它具有以下性质：  
&emsp;对于任意节点n，其左子树中的所有节点的值都小于n的值；  
&emsp;对于任意节点n，其右子树中的所有节点的值都大于n的值；  
&emsp;左子树和右子树也分别是二叉搜索树。  
这些性质使得二叉搜索树非常适合用于需要快速查找、插入和删除操作的应用场景。由于每个节点的值与它的左右子树中的所有节点值都有明确的关系，  
所以可以在O(h)的时间复杂度内完成查找、插入或删除操作，其中h是树的高度。在最坏情况下（例如树完全不平衡），时间复杂度可能会退化到O(n)，  
但在平均情况下，如果树保持平衡，那么这些操作的时间复杂度接近于O(log n)。  
  
AVL树（Adelson-Velsky and Landis Tree）是一种自平衡二叉搜索树。在AVL树中，任何节点的两个子树的高度最大差别为1，因此它也被称为高度平衡树。  
AVL树通过确保树的高度保持对数级别（O(log n)），从而保证了查找、插入和删除操作的时间复杂度不会超过O(log n)。  
  
AVL树是通过旋转操作来维持平衡的。当插入或删除一个节点导致树失去平衡时，会执行一种或多种旋转操作（单旋转或双旋转）以恢复平衡。这些旋转包括：  
&emsp;左旋转（Left Rotation）：用于修正右重的情况。  
&emsp;右旋转（Right Rotation）：用于修正左重的情况。  
&emsp;左右旋转（Left-Right Rotation）：先对左子树进行左旋转，然后对根节点进行右旋转，用于修正左子树右重的情况。  
&emsp;右左旋转（Right-Left Rotation）：先对右子树进行右旋转，然后对根节点进行左旋转，用于修正右子树左重的情况。  
  
代码分析：  
inorder_traversal.py文件实现了二叉树的中序递归遍历。  
```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
```
类TreeNode代表树的一个节点，value是节点的值，left是节点的左子树，right是节点的右子树。  
```python
def inorder_traversal(node):
    if node is not None:
        inorder_traversal(node.left)  # Traverse the left subtree
        print(node.value)  # Visit the root
        inorder_traversal(node.right)  # Traverse the right subtree
```
inorder_traversal函数实现了树的递归中序遍历，即先访问节点的左子树，再访问节点，最后访问节点的右子树。  
可以这样理解：  
1、无论二叉树有多大，都有一个根节点，把其整个左子树抽象成一个节点，其整个右子树抽象成一个节点。注意，抽象节点可能是空节点。  
2、对3个节点（1个真正节点，2个抽象节点），如果是真正的节点，直接打印其值。如果是抽象节点，调用inorder_traversal函数。当抽象节点是空节点时，inorder_traversal函数什么都不做。  
3、因为是中序遍历，所以先对左抽象节点调用inorder_traversal函数，再打印根节点的值，最后对右抽象节点调用inorder_traversal函数。  
4、因为每个抽象节点其实也是二叉树，所以，对每个抽象节点，重复1, 2, 3步骤。  
  
inorder_traversal_non_recursive.py文件实现了二叉树的非递归中序遍历。  
```python
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
```
1、从根节点开始，沿着left指针，一路走到最左边的节点。沿途经过的节点压入堆栈stack中。  
2、到达最左边的节点后（我们记作left_end），打印其值。  
3、如果left_end节点存在右子树，对右子树重复整个过程。  
4、left_end节点的子树都遍历完成后，则从stack中弹出left_end节点的父节点，打印其值，然后对其右子树重复整个过程。  
  
preorder_traversal_non_recursive.py文件实现了二叉树的前序非递归遍历。  
```python
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
```
1、打印根节点的值。如果根结点有右子节点，则将右子节点入栈。如果根节点有左子节点，则将左子节点入栈。  
2、从堆栈中弹出第一个元素，重复步骤1。直到堆栈为空。  
  
### License  
  
MIT
