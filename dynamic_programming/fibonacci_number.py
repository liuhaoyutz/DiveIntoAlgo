def fibonacci_bottom_up(n):
    """
    Bottom-up dynamic programming approach to compute the nth Fibonacci number.
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

def fibonacci_top_down(n, memo=None):
    """
    Top-down dynamic programming approach with memoization to compute the nth Fibonacci number.
    """
    if memo is None:
        memo = {}
        
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_top_down(n - 1, memo) + fibonacci_top_down(n - 2, memo)
    return memo[n]

def main():
    print("Welcome to the Fibonacci Calculator!")
    while True:
        try:
            method = input("Choose a method (1. Bottom-up 2. Top-down) or press 'q' to quit: ").strip().lower()
            if method == 'q':
                break
            elif method not in ['1', '2']:
                raise ValueError("Invalid choice, please try again.")

            n = int(input("Enter the index of the Fibonacci number you want to calculate (non-negative integer): "))
            if n < 0:
                raise ValueError("Please enter a non-negative integer.")

            if method == '1':
                result = fibonacci_bottom_up(n)
                print(f"Using bottom-up method, Fibonacci({n}) = {result}")
            else:  # method == '2'
                result = fibonacci_top_down(n)
                print(f"Using top-down method, Fibonacci({n}) = {result}")

        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print("-" * 40)

if __name__ == "__main__":
    main()
