class HashTable:
    def __init__(self, size=10):
        # Initialize the hash table with empty bucket list entries.
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def _hash(self, key):
        # A simple hash function that uses Python's built-in hash() function
        return hash(key) % self.size

    def put(self, key, value):
        # Insert or update the value associated with the key
        hash_index = self._hash(key)
        bucket = self.table[hash_index]
        
        found = False
        for index, (k, v) in enumerate(bucket):
            if k == key:
                bucket[index] = (key, value)  # Update existing key-value pair
                found = True
                break
        
        if not found:
            bucket.append((key, value))  # Add new key-value pair

    def get(self, key):
        # Retrieve the value associated with the key
        hash_index = self._hash(key)
        bucket = self.table[hash_index]
        
        for k, v in bucket:
            if k == key:
                return v  # Return the value if key is found
        
        raise KeyError(f"Key '{key}' not found")

    def remove(self, key):
        # Remove the key-value pair from the hash table
        hash_index = self._hash(key)
        bucket = self.table[hash_index]
        
        for index, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[index]  # Remove the key-value pair
                return
        
        raise KeyError(f"Key '{key}' not found")

    def __str__(self):
        # String representation of the hash table
        pairs = []
        for bucket in self.table:
            for key, value in bucket:
                pairs.append(f"{key}: {value}")
        return "{" + ", ".join(pairs) + "}"

# Example usage:
if __name__ == "__main__":
    ht = HashTable()

    # Put some key-value pairs into the hash table
    ht.put("apple", "fruit")
    ht.put("carrot", "vegetable")
    ht.put("banana", "fruit")

    print(ht)  # Output: {apple: fruit, carrot: vegetable, banana: fruit}

    # Get the value for a key
    print(ht.get("apple"))  # Output: fruit

    # Update an existing key's value
    ht.put("apple", "a type of fruit")
    print(ht.get("apple"))  # Output: a type of fruit

    # Remove a key-value pair
    ht.remove("carrot")
    print(ht)  # Output: {apple: a type of fruit, banana: fruit}
