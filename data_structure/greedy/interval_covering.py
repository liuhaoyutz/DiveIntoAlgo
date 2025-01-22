def min_intervals_to_cover(intervals, a, b):
    intervals.sort(key=lambda x: x[1])  # Sort by end points
    
    cover = []
    current_end = a
    
    while current_end < b:
        farthest = -1
        next_interval = None
        
        for interval in intervals:
            if interval[0] <= current_end and interval[1] > farthest:
                farthest = interval[1]
                next_interval = interval
        
        if not next_interval:
            return None  # Cannot cover the entire segment
        
        cover.append(next_interval)
        current_end = farthest
    
    return cover

# Example usage:
intervals = [(1, 3), (2, 5), (3, 7), (6, 9)]
cover = min_intervals_to_cover(intervals, 1, 9)
if cover:
    print("Intervals used to cover:", cover)
else:
    print("Cannot cover the entire segment.")
