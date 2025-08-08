def balance_workload(tasks, team_members):
    """
    Simple round-robin workload distribution.
    """
    assignments = {member: [] for member in team_members}
    i = 0
    for task in tasks:
        member = team_members[i % len(team_members)]
        assignments[member].append(task)
        i += 1
    return assignments
