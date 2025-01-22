# 回溯

回溯算法（Backtracking Algorithm）是一种通过构建所有可能的候选解并逐步探索这些候选解来解决问题的方法。  
它尝试一个候选解，如果发现这个候选解不满足问题的约束条件，或者在进一步扩展后无法得到可行解，则会“回溯”到上一步，并尝试其他可能的选择。  
  
find_path.py文件使用回溯算法找出所有从起点到终点的迷宫路径。  
```python
def find_paths(maze, start, end):
    """
    Use backtracking to find all paths from start to end in a maze.
    
    :param maze: 2D list representing the maze where 0 is an open path and 1 is a wall.
    :param start: Tuple (x, y) representing the starting position.
    :param end: Tuple (x, y) representing the ending position.
    :return: List of all paths from start to end.
    """
    paths = []

    def backtrack(x, y, path):
        # Check if current position is outside the maze or it's a wall or already visited
        if x < 0 or y < 0 or x >= len(maze) or y >= len(maze[0]) or maze[x][y] == 1 or (x, y) in path:
            return
        
        # Add current position to the path
        path.append((x, y))
        
        # If we reach the end position, add this path to the result
        if (x, y) == end:
            paths.append(list(path))
        else:
            # Try moving in all possible directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                backtrack(x + dx, y + dy, path)
        
        # Backtrack: remove the current position from the path before returning
        path.pop()

    backtrack(start[0], start[1], [])
    return paths

# Example usage:
if __name__ == "__main__":
    maze_example = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start_position = (0, 0)
    end_position = (4, 4)
    print("All paths from start to end:")
    for path in find_paths(maze_example, start_position, end_position):
        print(path)
```
find_paths 函数接收一个迷宫（用二维列表表示，其中0表示可以通行的道路，1表示障碍物），起始点和终点。  
它定义了一个内部函数 backtrack 来执行实际的回溯逻辑。每当找到一条到达终点的路径时，就将其添加到结果列表中。  
如果当前位置不是终点，函数将继续尝试所有四个方向的移动，直到所有可能性都被探索完毕。  
最后，它会通过path.pop()语句移除当前位置，即回溯，以便可以尝试其他路径。  
  
n_queens.py文件利用回溯算法解决n皇后问题。  
```python
def solve_n_queens(n):
    """
    Solve the N-Queens puzzle using backtracking.

    :param n: The number of queens and the size of the chessboard (n x n).
    :return: A list of solutions, where each solution is represented as a list of strings.
             Each string represents a row on the chessboard with 'Q' for a queen and '.' for an empty space.
    """
    def backtrack(row=0):
        if row == n:
            # Found a solution, add it to the output
            board = ['.' * i + 'Q' + '.' * (n - i - 1) for i in queens]
            result.append(board)
            return
        
        for col in range(n):
            if cols[col] or hills[row - col] or dales[row + col]:
                continue
            
            # Place a queen and mark its columns and diagonals as under attack
            queens.append(col)
            cols[col] = True
            hills[row - col] = True
            dales[row + col] = True

            # Move on to the next row
            backtrack(row + 1)

            # Backtrack and try the next column
            queens.pop()
            cols[col] = False
            hills[row - col] = False
            dales[row + col] = False

    cols = [False] * n  # Track columns under attack
    hills = [False] * (2 * n - 1)  # Track \ diagonals under attack
    dales = [False] * (2 * n - 1)  # Track / diagonals under attack
    queens = []  # Keep track of the positions of the queens (only column indices)
    result = []  # Store the final solutions

    backtrack()
    return result

# Example usage:
if __name__ == "__main__":
    n = 4  # Size of the chessboard and the number of queens
    solutions = solve_n_queens(n)
    print(f"Found {len(solutions)} solutions for the {n}-Queens puzzle:")
    for solution in solutions:
        for row in solution:
            print(row)
        print("-" * n)
```
cols: 用于跟踪哪些列已经被占用，共有n列。  
hills: 用于跟踪左斜线（从左上到右下）上的位置是否被占用，共有2*n-1条左斜线。  
dales: 用于跟踪右斜线（从右上到左下）上的位置是否被占用，共有2*n-1条右斜线。  
queens: 用来记录每一行中皇后所在的列索引。只保存列索引，行索引是隐含的。例如[1, 3, 0, 2]表示皇后在(0, 1), (1, 3), (2, 0), (3, 2)位置。  
result: 存储最终所有解决方案的列表。  
  
backtrack函数中，如果row等于n，说明找到了一个方案，将棋盘布局保存到result中返回。  
对于每一行，遍历所有列。  
如果cols[col]为true，说明这一列已经有皇后。如果hills[row - col]为true，说明左斜线上已经有皇后，如果dales[row + col]为true，说明右斜线上已经有皇后。  
如果没有冲突，则将col保存到queens，记录皇后所在的列索引。同时将cols[col]，hills[row - col]，dales[row + col]设置true，表示这些列，左斜线，右斜线上已经有皇后。  
以row+1为参数，递归调用backtrack，处理下一行。  
回溯：一旦递归返回（无论是因为找到了解还是因为无法继续放置），执行queens.pop()，移除最后放置的皇后，并尝试该行的下一列。注意在此之前，将cols, hills, dales设置为False。  
  
### License  
  
MIT
