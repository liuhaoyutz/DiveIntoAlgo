def activity_selection(start, finish):
    # Sort activities based on their finish time
    activities = sorted(zip(finish, start))
    selected_activities = [activities[0]]
    
    for i in range(1, len(activities)):
        if activities[i][1] >= selected_activities[-1][0]:  # Start time is after the last finish time
            selected_activities.append(activities[i])
    
    return [(act[1], act[0]) for act in selected_activities]

# Example usage:
start_times = [1, 3, 0, 5, 8, 5]
finish_times = [2, 4, 6, 7, 9, 9]
selected = activity_selection(start_times, finish_times)
print("Selected activities:", selected)
