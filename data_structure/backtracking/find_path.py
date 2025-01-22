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
