import array

# Initialize array (only store numerical data of the same type)
my_array = array.array('i', [1, 2, 3, 4, 5])  # 'i' stands for signed integer
print("Initialized array:", my_array)

# Access element (index starts from 0)
element = my_array[2]  # Access the third element
print("Access element at index 2:", element)

# Insert element (at a specified position)
my_array.insert(2, 100)  # Insert new element at index 2
print("After inserting 100 at index 2:", my_array)

# Delete element (by index)
del my_array[0]  # Delete the element at index 0
print("After deleting element at index 0:", my_array)

# Traverse the array
print("Traversing the array:")
for item in my_array:
    print(item)

# Search for an element (check existence and get index)
try:
    index = my_array.index(3)  # Return index if found
    print("Found 3 at index:", index)
except ValueError:
    print("3 not found in array")

# Expand the array (add new elements)
my_array.append(6)  # Add a single element at the end of the array
print("After appending 6:", my_array)
my_array.extend([7, 8, 9])  # Add multiple elements at the end of the array
print("After extending with [7, 8, 9]:", my_array)
