# Critical Chain Planner - User Tutorial

This tutorial will guide you through using the Critical Chain Project Management (CCPM) Planner.

## Introduction

The Critical Chain Planner helps you plan and manage projects using the Critical Chain Project Management methodology. This approach focuses on resource constraints and buffer management to ensure project success.

## Getting Started

1. Launch the application by running:
   ```
   python ccpm_Planner.py
   ```

2. The application opens with four tabs:
   - **Tasks**: Manage project tasks
   - **Resources**: Manage project resources
   - **Gantt Chart**: View project schedule as a Gantt chart
   - **Fever Chart**: Track project buffer consumption

3. The application comes pre-loaded with a sample project for demonstration.

## Tutorial: Creating a Simple Project

Let's walk through creating a simple software development project:

### Step 1: Define Resources

1. Click on the **Resources** tab.
2. Review existing resources or add new ones by clicking "Add Resource".
3. Let's add a new resource:
   - Click "Add Resource"
   - Resource ID: "BA1"
   - Resource Name: "Business Analyst"
   - Availability: 100%
   - Click "Save"

### Step 2: Define Tasks

1. Click on the **Tasks** tab.
2. Click "Add Task" to create a new task:
   - Task ID: 8
   - Task Name: "Requirements Gathering"
   - Duration: 5
   - Resources: "BA1"
   - Predecessors: (leave empty as this is the first task)
   - Completion: 0%
   - Click "Save"

3. Add another task:
   - Task ID: 9
   - Task Name: "User Interface Design"
   - Duration: 8
   - Resources: "R1"
   - Predecessors: "8"
   - Completion: 0%
   - Click "Save"

4. Add a final task:
   - Task ID: 10
   - Task Name: "Integration Testing"
   - Duration: 6
   - Resources: "R3"
   - Predecessors: "9"
   - Completion: 0%
   - Click "Save"

### Step 3: Schedule the Project

1. Click "Schedule Tasks" button on the Tasks tab.
2. A confirmation message will appear when scheduling is complete.

### Step 4: View the Gantt Chart

1. Click on the **Gantt Chart** tab.
2. The chart shows:
   - All tasks as horizontal bars
   - Critical tasks in red
   - Non-critical tasks in blue
   - The project buffer in green at the end of the critical chain

3. If you don't see your new tasks, click "Update Gantt Chart" to refresh.

### Step 5: Simulate Project Progress

1. Return to the **Tasks** tab.
2. Select a task and click "Edit Task".
3. Update the "Completion" percentage to simulate progress.
4. Repeat for other tasks.
5. Click "Schedule Tasks" to update the project schedule.

### Step 6: Monitor Buffer Consumption

1. Go to the **Fever Chart** tab.
2. The chart shows buffer consumption against project completion:
   - Green zone: Safe
   - Yellow zone: Warning
   - Red zone: Danger

3. Click "Update Fever Chart" to refresh after making changes.

## Advanced Usage

### Managing Task Dependencies

1. Tasks can have multiple predecessors, separated by commas (e.g., "1,3,5").
2. When you remove a task, the Planner automatically updates predecessors in remaining tasks.

### Resource Leveling

The Planner automatically resolves resource conflicts by:
1. Identifying tasks that require the same resource at the same time
2. Delaying lower-priority tasks to avoid conflicts

### Critical Chain Identification

The critical chain is the longest path through the project, considering both:
1. Task dependencies (like the traditional critical path)
2. Resource constraints (unique to critical chain methodology)

Tasks on the critical chain are highlighted in red on the Gantt chart.

### Project Buffer

The project buffer is automatically calculated as 50% of the critical chain duration and added to the end of the project. This buffer protects the project delivery date against variations in task durations.

## Common Scenarios

### Adding a New Phase to the Project

1. Add new tasks with appropriate predecessors
2. Assign resources
3. Click "Schedule Tasks" to update the project schedule

### Handling Resource Changes

1. Go to the **Resources** tab
2. Edit or add resources as needed
3. Update task assignments on the **Tasks** tab
4. Re-schedule the project

### Dealing with Task Delays

1. Update task completion percentages to reflect actual progress
2. Reschedule the project
3. Monitor buffer consumption on the fever chart
4. Take action if buffer consumption enters the yellow or red zones

## Troubleshooting

### Tasks Not Appearing in the Correct Order

**Issue**: Tasks may not be scheduled as expected.
**Solution**: Check predecessor relationships and ensure there are no circular dependencies.

### Resource Conflicts Not Resolved

**Issue**: Two tasks using the same resource still scheduled simultaneously.
**Solution**: Ensure resources are correctly assigned to tasks and re-schedule.

### Application Crashes When Scheduling

**Issue**: Application crashes during scheduling.
**Solution**: Check for circular dependencies in task relationships or invalid input data.

## Conclusion

The Critical Chain Planner provides a visual and interactive way to learn and apply Critical Chain Project Management principles. By focusing on resource constraints and buffer management, you can create more realistic project schedules and effectively track project progress.

Happy project planning!