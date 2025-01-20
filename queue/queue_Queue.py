import queue

# Create a thread-safe queue
q = queue.Queue()

# Put elements into the queue
for i in range(5):
    q.put(i)

print("Initial queue size:", q.qsize())

# Get and remove elements from the queue
while not q.empty():
    print("Dequeued element:", q.get())

print("Is the queue empty now?", q.empty())
