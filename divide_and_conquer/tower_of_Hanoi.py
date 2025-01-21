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
