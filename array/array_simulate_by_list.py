# Initialize list (simulate array)
my_list = [1, 2, 3, 4, 5]
print("Initialized list:", my_list)

# Access element (index starts from 0)
element = my_list[2]  # Access the third element
print("Access element at index 2:", element)

# Insert element (at a specified position)
my_list.insert(2, 'inserted')  # Insert new element at index 2
print("After inserting element at index 2:", my_list)

# Delete element (by value or index)
my_list.remove('inserted')  # Remove the first occurrence of the value 'inserted'
print("After removing 'inserted':", my_list)
del my_list[0]  # Delete the element at index 0
print("After deleting element at index 0:", my_list)

# Traverse the list
print("Traversing the list:")
for item in my_list:
    print(item)

# Search for an element (check existence and get index)
if 3 in my_list:
    print("Found 3 at index:", my_list.index(3))

# Expand the list (add new elements)
my_list.append(6)  # Add a single element at the end of the list
print("After appending 6:", my_list)
my_list.extend([7, 8, 9])  # Add multiple elements at the end of the list
print("After extending with [7, 8, 9]:", my_list)

# Lists are dynamic, so explicit expansion is not necessary
