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