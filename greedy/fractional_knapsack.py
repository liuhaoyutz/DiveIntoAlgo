def fractional_knapsack(weights, values, capacity):
    # Create a list of tuples (value/weight, weight, value)
    items = [(v/w, w, v) for w, v in zip(weights, values)]
    items.sort(reverse=True, key=lambda item: item[0])  # Sort by value per unit weight
    
    total_value = 0
    remaining_capacity = capacity
    
    for value_per_weight, weight, value in items:
        if remaining_capacity == 0:
            break
        
        take_amount = min(weight, remaining_capacity)
        total_value += take_amount * value_per_weight
        remaining_capacity -= take_amount
    
    return total_value

# Example usage:
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
max_value = fractional_knapsack(weights, values, capacity)
print(f"Maximum value in knapsack: {max_value}")
