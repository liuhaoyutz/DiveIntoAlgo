# Create a hash table (dictionary)
hash_table = {}

# Insert key-value pairs
hash_table['apple'] = 'fruit'
hash_table['banana'] = 'fruit'
hash_table['carrot'] = 'vegetable'

# Look up the value associated with a key
print(hash_table['apple'])  # Output: fruit

# Update the value for an existing key
hash_table['apple'] = 'a type of fruit'
print(hash_table['apple'])  # Output: a type of fruit

# Delete a key-value pair
del hash_table['banana']

# Check if a key exists in the hash table
print('banana' in hash_table)  # Output: False

# Iterate over the hash table
for key, value in hash_table.items():
    print(f"{key}: {value}")

# Get the size of the hash table
print(len(hash_table))  # Output: the current number of key-value pairs in the hash table
