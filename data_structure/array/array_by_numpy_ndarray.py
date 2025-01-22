import numpy as np

# Initialize array
my_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)  # Create a 1D array with elements of type 32-bit integer
print("Initialized array:", my_array)

# Access element (index starts from 0)
element = my_array[2]  # Access the third element
print("Access element at index 2:", element)

# Insert element (NumPy arrays are fixed-size; insertion typically requires creating a new array)
# Here we use np.insert to insert an element at a specified position
inserted_array = np.insert(my_array, 2, 100)  # Insert value 100 at index 2
print("After inserting 100 at index 2:", inserted_array)

# Delete element (similarly, deletion typically requires creating a new array)
# Use np.delete to remove an element at a specified index
deleted_array = np.delete(inserted_array, 0)  # Remove the element at index 0
print("After deleting element at index 0:", deleted_array)

# Traverse the array
print("Traversing the array:")
for item in deleted_array:
    print(item)

# Search for an element (check existence and get index)
# Use np.where to find the indices where the condition is met
indices = np.where(deleted_array == 3)[0]
if indices.size > 0:
    print("Found 3 at index:", indices[0])
else:
    print("3 not found in array")

# Expand the array (add new elements)
# You can use np.append or np.concatenate to add elements
expanded_array = np.append(deleted_array, [6])  # Add a single element at the end
print("After appending 6:", expanded_array)
expanded_array = np.concatenate((expanded_array, [7, 8, 9]))  # Add multiple elements at the end
print("After extending with [7, 8, 9]:", expanded_array)

# Modify array size (reshape)
reshaped_array = expanded_array.reshape((3, 3))  # Convert the 1D array into a 2D array with shape (3, 3)
print("Reshaped array to 3x3 matrix:\n", reshaped_array)

# Note: NumPy arrays are fixed-size, so insertion and deletion operations create new array objects
