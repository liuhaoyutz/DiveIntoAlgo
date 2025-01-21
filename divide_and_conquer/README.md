# 分治

分治（Divide and Conquer）是一种多路归并的递归算法。它将一个复杂的问题分解为若干个规模较小、相互独立且与原问题类型相同的子问题，  
递归地求解这些子问题，然后将各子问题的解合并得到原问题的解。  
  
quick_sort.py文件实现的快速排序是一个经典的分治算法例子。  
```python
def quick_sort(arr):
    # If the array length is less than or equal to 1, it is already sorted, return it directly.
    if len(arr) <= 1:
        return arr
    
    # Choose a pivot element. Here we choose the first element of the list.
    pivot = arr[0]
    
    # Create two lists: one for elements less than or equal to the pivot,
    # and another for elements greater than the pivot.
    less_than_pivot = [x for x in arr[1:] if x <= pivot]
    greater_than_pivot = [x for x in arr[1:] if x > pivot]
    
    # Recursively call quick_sort on both sublists and combine the results.
    # The less_than_pivot sublist comes before the pivot, which comes before the greater_than_pivot sublist.
    return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)
```
  
build_binary_tree.py文件实现利用分治法创建二叉树。  
当我们有二叉树的中序和前序遍历结果，或者有中序或后序遍历结果，可以用分治法来构建二叉树。  
这是因为前序和后序遍历可以告诉我们根节点的信息，而中序遍历可以用来确定左子树和右子树的范围。  
```python
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
```
buildTree函数接收2个参数，第一个是前序遍历列表，第二个参数是中序遍历列表。  
1、前序遍历列表的第1个元素就是二叉树的根节点root。  
2、取得二叉树根节点在中序遍历列表中的索引。  
3、根节点将中序遍历列表分成左、右两部分，分别是左子树的中序遍历列表left_inorder和右子树的中序遍历列表right_inorder。注意，根节点被排除在外。  
4、由left_inorder列表的长度可知左子树的节点个数。由此可在前序遍历列表中取出左子树的前序遍历列表left_preorder和右子树的前序遍历列表right_preorder。  
5、以左子树的前序遍历列表和中序遍历列表为参数，递归调用buildTree，返回值赋值给root.left。  
6、以右子树的前序遍历列表和中序遍历列表为参数，递归调用buildTree，返回值赋值给root.right。  
7、返回root。  
  
考虑最简单的例子：  
只有一个节点1。前序遍历列表是[1]，中序遍历列表也是[1]。  
第一层调用buildTree时，left_preorder, left_inorder，right_perorder和right_inorder都是None。  
第二层调用buildTree时，返回None。回到第一层buildTree，root.left为None，root.right为None。二叉树构建完毕，只有一个root节点。  
  
tower_of_Hanoi.py文件用分治法解决汉诺塔问题。  
```python
def hanoi_tower(n, source, auxiliary, target):
    """
    Solve the Tower of Hanoi puzzle using divide and conquer.

    :param n: Number of disks.
    :param source: The name of the source peg (e.g., 'A').
    :param auxiliary: The name of the auxiliary peg (e.g., 'B').
    :param target: The name of the target peg (e.g., 'C').
    """
    if n == 1:
        # Base case: only one disk, move it from source to target
        print(f"Move disk 1 from {source} to {target}.")
        return

    # Step 1: Move n-1 disks from source to auxiliary, so they are out of the way
    hanoi_tower(n - 1, source, target, auxiliary)

    # Step 2: Move the nth disk from source to target
    print(f"Move disk {n} from {source} to {target}.")

    # Step 3: Move the n-1 disks that we left on auxiliary to target
    hanoi_tower(n - 1, auxiliary, source, target)

# Example usage:
if __name__ == "__main__":
    num_disks = 3  # Number of disks in the Tower of Hanoi puzzle
    print(f"Solving the Tower of Hanoi with {num_disks} disks:")
    hanoi_tower(num_disks, 'A', 'B', 'C')
```
  
### License  
  
MIT
