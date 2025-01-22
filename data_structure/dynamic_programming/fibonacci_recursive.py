def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Example usage:
if __name__ == "__main__":
    n = 3  # Replace with the desired index of the Fibonacci sequence
    print(f"Fibonacci({n}) = {fibonacci_recursive(n)}")
