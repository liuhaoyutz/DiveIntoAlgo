class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def insert_key(self, k):
        self.heap.append(k)
        i = len(self.heap) - 1
        self._heapify_up(i)

    def _heapify_up(self, i):
        while i != 0 and self.heap[self.parent(i)] > self.heap[i]:
            # Swap this node with its parent
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)

    def extract_min(self):
        if len(self.heap) <= 0:
            return float("inf")
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def _heapify_down(self, i):
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self._heapify_down(smallest)

# Example usage:
if __name__ == "__main__":
    min_heap = MinHeap()
    elements = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    for element in elements:
        min_heap.insert_key(element)

    print("Extracted minimums:")
    while True:
        min_val = min_heap.extract_min()
        if min_val == float("inf"):
            break
        print(min_val)
