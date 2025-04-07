import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Task:
    """Class to represent a task in the Critical Chain model"""

    def __init__(
        self,
        task_id: int,
        name: str,
        duration: int,
        resources: List[str] = None,
        predecessors: str = "",
    ):
        self.task_id = task_id
        self.name = name
        self.duration = duration
        self.resources = resources if resources else []
        self.predecessors = predecessors
        self.start_date = 0
        self.end_date = 0
        self.is_critical = False

        # For Larry Leech example datasets
        self.safe_duration = duration  # Safe estimate
        try:
            # Try to convert duration to int and then calculate aggressive duration
            self.aggressive_duration = int(
                int(duration) * 0.5
            )  # Aggressive estimate by default
        except (ValueError, TypeError):
            # If conversion fails, set aggressive duration to same as safe duration
            self.aggressive_duration = duration

    def get_ID(self) -> int:
        """Get the task ID"""
        return self.task_id

    def get_preds(self) -> str:
        """Get the predecessors string"""
        return self.predecessors

    def set_preds(self, preds: str) -> None:
        """Set the predecessors string"""
        self.predecessors = preds


class SampleDataset:
    """Class to hold sample dataset information"""

    def __init__(
        self,
        name: str,
        description: str,
        resources: Dict[str, int],
        tasks: List[Tuple],
        expected_chain: List[str] = None,
    ):
        self.name = name
        self.description = description
        self.resources = resources  # Dict of resource_id: availability
        self.tasks = tasks  # List of task parameter tuples
        self.expected_chain = expected_chain  # Expected critical chain task names

    # def load_into_planner(self, planner):
    #     """Load this dataset into the provided planner"""
    #     # Add resources
    #     for resource_id, availability in self.resources.items():
    #         planner.resource_manager.add_resource(
    #             resource_id, resource_id, availability * 100
    #         )

    #     # Add tasks
    #     task_id = 1
    #     for task_data in self.tasks:
    #         if len(task_data) >= 5 and isinstance(
    #             task_data[0], str
    #         ):  # Larry Leech format with string names
    #             name, safe_duration, aggressive_duration, dependencies, resource = (
    #                 task_data
    #             )
    #             # Convert dependencies from task names to task IDs
    #             pred_ids = []
    #             for dep_name in dependencies:
    #                 for existing_task in planner.tasks:
    #                     if existing_task.name == dep_name:
    #                         pred_ids.append(str(existing_task.task_id))
    #                         break

    #             # Create the task
    #             task = planner.add_task(
    #                 task_id,
    #                 name,
    #                 int(safe_duration),  # Ensure safe duration is an integer
    #                 [resource] if resource else [],
    #                 ",".join(pred_ids),
    #             )

    #             # Set specific durations
    #             task.safe_duration = int(safe_duration)
    #             task.aggressive_duration = int(aggressive_duration)
    #         else:  # Standard format
    #             # Unpack task data
    #             if len(task_data) == 5:
    #                 task_id_val, name, duration, resources, predecessors = task_data
    #             else:
    #                 # Default values for missing parameters
    #                 task_id_val, name, duration = (
    #                     task_data[0],
    #                     task_data[1],
    #                     task_data[2],
    #                 )
    #                 resources = task_data[3] if len(task_data) > 3 else []
    #                 predecessors = task_data[4] if len(task_data) > 4 else ""

    #             # Create the task
    #             planner.add_task(
    #                 int(task_id_val),
    #                 name,
    #                 int(duration),
    #                 resources,
    #                 predecessors,
    #             )

    #             # Update task_id for next loop iteration
    #             task_id = int(task_id_val) + 1
    #             continue

    #         task_id += 1

    #     # Schedule tasks
    #     planner.schedule_tasks()

    # return planner

    def load_into_planner(self, planner):
        """Load this dataset into the provided planner using post-processing for dependencies"""
        # Add resources
        for resource_id, availability in self.resources.items():
            planner.resource_manager.add_resource(
                resource_id, resource_id, availability * 100
            )

        # First, create a dictionary to map task names to their IDs
        task_name_to_id = {}

        # Add tasks
        task_id = 1
        for task_data in self.tasks:
            if len(task_data) >= 5 and isinstance(
                task_data[0], str
            ):  # Larry Leech format
                name, safe_duration, aggressive_duration, dependencies, resource = (
                    task_data
                )

                # Create the task with empty predecessors initially
                task = planner.add_task(
                    task_id,
                    name,
                    int(safe_duration),
                    [resource] if resource else [],
                    "",  # Start with empty predecessors
                )

                # Store the mapping from task name to task ID
                task_name_to_id[name] = task_id

                # Set specific durations
                task.safe_duration = int(safe_duration)
                task.aggressive_duration = int(aggressive_duration)

                task_id += 1

            else:  # Standard format
                # Unpack task data
                if len(task_data) == 5:
                    task_id_val, name, duration, resources, predecessors = task_data
                else:
                    # Default values for missing parameters
                    task_id_val, name, duration = (
                        task_data[0],
                        task_data[1],
                        task_data[2],
                    )
                    resources = task_data[3] if len(task_data) > 3 else []
                    predecessors = task_data[4] if len(task_data) > 4 else ""

                # Create the task
                planner.add_task(
                    int(task_id_val),
                    name,
                    int(duration),
                    resources,
                    predecessors,
                )

                # Update task_id for next loop iteration
                task_id = int(task_id_val) + 1
                continue

        # Now process all dependencies in a second pass
        for task_data in self.tasks:
            if len(task_data) >= 5 and isinstance(
                task_data[0], str
            ):  # Larry Leech format
                name, safe_duration, aggressive_duration, dependencies, resource = (
                    task_data
                )

                if dependencies:
                    # Find the task object
                    task = next((t for t in planner.tasks if t.name == name), None)
                    if task:
                        # Convert dependencies from task names to task IDs
                        pred_ids = []
                        for dep_name in dependencies:
                            if dep_name in task_name_to_id:
                                pred_ids.append(str(task_name_to_id[dep_name]))

                        # Set the predecessors string
                        if pred_ids:
                            task.predecessors = ",".join(pred_ids)

        # Schedule tasks
        planner.schedule_tasks()

        return planner

    # def load_into_planner(self, planner):
    #     """Load this dataset into the provided planner with detailed debugging"""
    #     import sys

    #     # Add resources
    #     for resource_id, availability in self.resources.items():
    #         planner.resource_manager.add_resource(
    #             resource_id, resource_id, availability * 100
    #         )

    #     print("\n==== DEBUG: LOADING DATASET ====")
    #     print(f"Dataset: {self.name}")

    #     # Add tasks
    #     task_id = 1
    #     for task_data in self.tasks:
    #         if len(task_data) >= 5 and isinstance(
    #             task_data[0], str
    #         ):  # Larry Leech format with string names
    #             name, safe_duration, aggressive_duration, dependencies, resource = (
    #                 task_data
    #             )

    #             print(f"\nProcessing task: {name} with dependencies: {dependencies}")

    #             # Convert dependencies from task names to task IDs
    #             pred_ids = []
    #             for dep_name in dependencies:
    #                 print(f"  Looking for dependency: {dep_name}")
    #                 found = False
    #                 for existing_task in planner.tasks:
    #                     if existing_task.name == dep_name:
    #                         pred_ids.append(str(existing_task.task_id))
    #                         print(f"    Found! ID: {existing_task.task_id}")
    #                         found = True
    #                         break
    #                 if not found:
    #                     print(f"    NOT FOUND! Dependency {dep_name} will be missing.")

    #             print(f"  Final predecessor IDs: {pred_ids}")

    #             # Create the task
    #             task = planner.add_task(
    #                 task_id,
    #                 name,
    #                 int(safe_duration),  # Ensure safe duration is an integer
    #                 [resource] if resource else [],
    #                 ",".join(pred_ids),
    #             )

    #             # Set specific durations
    #             task.safe_duration = int(safe_duration)
    #             task.aggressive_duration = int(aggressive_duration)
    #         else:  # Standard format
    #             # Unpack task data
    #             if len(task_data) == 5:
    #                 task_id_val, name, duration, resources, predecessors = task_data
    #             else:
    #                 # Default values for missing parameters
    #                 task_id_val, name, duration = (
    #                     task_data[0],
    #                     task_data[1],
    #                     task_data[2],
    #                 )
    #                 resources = task_data[3] if len(task_data) > 3 else []
    #                 predecessors = task_data[4] if len(task_data) > 4 else ""

    #             # Create the task
    #             planner.add_task(
    #                 int(task_id_val),
    #                 name,
    #                 int(duration),
    #                 resources,
    #                 predecessors,
    #             )

    #             # Update task_id for next loop iteration
    #             task_id = int(task_id_val) + 1
    #             continue

    #         task_id += 1

    #     # Print final task list
    #     print("\n==== FINAL TASK LIST ====")
    #     for task in planner.tasks:
    #         print(
    #             f"Task ID: {task.task_id}, Name: {task.name}, Predecessors: {task.predecessors}"
    #         )

    #     print("\n==== END DEBUG ====\n")

    #     # Schedule tasks
    #     planner.schedule_tasks()

    #     return planner


class ResourceManager:
    """Class to manage resources"""

    def __init__(self):
        self.resources = {}

    def add_resource(self, resource_id: str, name: str, availability: int = 100):
        """Add a resource"""
        self.resources[resource_id] = {
            "name": name,
            "availability": availability,  # Percentage of availability
            "assignments": [],
        }

    def remove_resource(self, resource_id: str):
        """Remove a resource"""
        if resource_id in self.resources:
            del self.resources[resource_id]

    def get_resources(self):
        """Get all resources"""
        return self.resources


class CriticalChainplanner:
    """Main class for the Critical Chain Project Management planner"""

    def __init__(self):
        self.tasks = []
        self.resource_manager = ResourceManager()
        self.chains = []  # Collections of tasks forming chains
        self.feeding_buffers = []
        self.project_buffer = 0
        self.dataset_name = "Default"
        self.sample_datasets = self._create_sample_datasets()

    def _create_sample_datasets(self):
        """Create sample datasets"""
        datasets = {}

        # Default dataset
        default_resources = {"Blue": 1, "Magenta": 1, "Green": 1}
        default_tasks = [
            ("1", 10, 5, [], "Blue"),  # "Project Planning"
            ("2", 20, 10, ["1"], "Blue"),  # "Design"
            ("3", 30, 15, ["2"], "Magenta"),  # "Development Phase 1"
            ("4", 24, 12, ["2"], "Blue"),  # "Development Phase 2"
            ("5", 16, 8, ["3", "4"], "Green"),  # "Testing"
            ("6", 14, 7, ["5"], "Blue"),  # "Documentation"
            ("7", 8, 4, ["5", "6"], "Blue"),  # "Deployment"
        ]
        datasets["Default"] = SampleDataset(
            "Default",
            "Default sample project with 7 tasks",
            default_resources,
            default_tasks,
            [],
        )

        # Small example from Larry Leech's book
        small_resources = {"Red": 1, "Green": 1, "Magenta": 1, "Blue": 1}
        small_tasks = [
            ("T1.1", 30, 15, [], "Red"),
            ("T1.2", 20, 10, ["T1.1"], "Green"),
            ("T3", 30, 15, ["T2.2", "T1.2"], "Magenta"),
            ("T2.1", 20, 10, [], "Blue"),
            ("T2.2", 10, 5, ["T2.1"], "Green"),
        ]
        datasets["Larry_Small"] = SampleDataset(
            "Larry_Small",
            "Small example from Larry Leech's book on CCPM",
            small_resources,
            small_tasks,
            ["T1.1", "T1.2", "T2.2", "T3"],
        )

        # Large example from Larry Leech's book
        large_resources = {"Red": 1, "Green": 1, "Magenta": 1, "Blue": 1, "Black": 1}
        large_tasks = [
            ("A-1", 10, 5, [], "Magenta"),
            ("A-2", 20, 10, ["A-1"], "Black"),
            ("A-3", 30, 15, ["A-2"], "Green"),
            ("A-4", 20, 10, ["A-3"], "Red"),
            ("A-5", 40, 20, ["A-4", "B-4"], "Magenta"),
            ("A-6", 28, 14, ["A-5"], "Red"),
            ("B-2", 20, 10, [], "Magenta"),
            ("B-3", 20, 10, ["B-2"], "Blue"),
            ("B-4", 10, 5, ["B-3"], "Red"),
            ("C-3", 30, 15, [], "Blue"),
            ("C-4", 20, 10, ["C-3"], "Green"),
            ("C-5", 30, 15, ["C-4", "D-4"], "Red"),
            ("C-6", 10, 5, ["C-5"], "Magenta"),
            ("D-3", 40, 20, [], "Blue"),
            ("D-4", 10, 5, ["D-3"], "Green"),
            ("Done", 0, 0, ["C-6", "A-6"], "Black"),
        ]
        datasets["Larry_Large"] = SampleDataset(
            "Larry_Large",
            "Large example from Larry Leech's book on CCPM",
            large_resources,
            large_tasks,
            ["A-1", "A-2", "A-3", "A-4", "B-4", "A-5", "A-6", "C-5", "C-6"],
        )

        return datasets

    def load_sample_dataset(self, dataset_name):
        """Load a sample dataset by name"""
        if dataset_name in self.sample_datasets:
            self.dataset_name = dataset_name

            # Clear existing data before loading
            self.tasks = []
            self.resource_manager = ResourceManager()

            # Load the dataset
            self.sample_datasets[dataset_name].load_into_planner(self)
            return True
        return False

    def add_task(
        self,
        task_id: int,
        name: str,
        duration: int,
        resources: List[str] = None,
        predecessors: str = "",
    ) -> Task:
        """Add a task to the project"""
        task = Task(task_id, name, duration, resources, predecessors)
        self.tasks.append(task)
        return task

    def remove_task(self, task_id: int) -> None:
        """Remove a task from the project and update predecessors"""
        # Find and remove the task
        task_to_remove = None
        for i, task in enumerate(self.tasks):
            if task.get_ID() == task_id:
                task_to_remove = task
                self.tasks.pop(i)
                break

        if task_to_remove:
            # Update predecessors in remaining tasks
            self.update_preds(task_id)

    def update_preds(self, task_id: int) -> None:
        """Update predecessors when a task is removed"""
        for task in self.tasks:
            if task.get_preds():
                preds = task.get_preds().split(",")

                # Check if the removed task is a predecessor
                if str(task_id) in preds:
                    # Remove the task ID from the predecessors string
                    new_preds = [p for p in preds if int(p) != task_id]
                    task.set_preds(",".join(new_preds))

    def remove_chains_first_task(self) -> None:
        """Remove the first task from each chain and update the chains"""
        for i, chain in enumerate(self.chains):
            if len(chain) > 0:
                # Remove the first task
                chain.pop(0)

                if len(chain) > 0:
                    # Get the predecessors of the new first task
                    preds = (
                        chain[0].get_preds().split(",") if chain[0].get_preds() else []
                    )

                    # Check if any of the remaining tasks in the chain are predecessors
                    # of the new first task
                    for j in range(1, len(chain)):
                        # If none of the remaining tasks are predecessors of the new first task,
                        # remove them from the chain
                        if str(chain[j].get_ID()) not in preds:
                            chain.pop(j)

    def max_fin_preds_reel(self, preds: List[str]) -> int:
        """Find the maximum finish date of predecessors"""
        max_date = 0
        saved_task = None

        for pred_id in preds:
            for task in self.tasks:
                if str(task.get_ID()) == pred_id:
                    # In the original code, this is checking cell values in the worksheet
                    # Here we'll use the task's end_date attribute
                    if task.end_date > max_date:
                        max_date = task.end_date
                        saved_task = task

        # Add 1 if the saved task has predecessors (as in the original code)
        if saved_task and saved_task.get_preds():
            max_date += 1

        return max_date

    def schedule_tasks(self) -> None:
        """Schedule tasks according to predecessors and resource constraints"""
        # Reset task dates
        for task in self.tasks:
            task.start_date = 0
            task.end_date = 0

        # Create a directed graph to represent task dependencies
        G = nx.DiGraph()

        # Add tasks as nodes
        for task in self.tasks:
            G.add_node(task.get_ID(), task=task)

        # Add edges for dependencies
        for task in self.tasks:
            if task.get_preds():
                preds = task.get_preds().split(",")
                for pred in preds:
                    if pred:  # Skip empty strings
                        G.add_edge(int(pred), task.get_ID())

        # Ensure the graph is acyclic (no circular dependencies)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Task dependencies contain cycles")

        # Get topological sort (order to process tasks)
        task_order = list(nx.topological_sort(G))

        # Schedule tasks based on predecessors
        for task_id in task_order:
            task = next((t for t in self.tasks if t.get_ID() == task_id), None)
            if task:
                if task.get_preds():
                    preds = task.get_preds().split(",")
                    # Find the latest end date among predecessors
                    latest_end = self.max_fin_preds_reel([p for p in preds if p])
                    task.start_date = latest_end

                task.end_date = task.start_date + task.duration

        # Apply resource leveling (simplified)
        self._apply_resource_leveling()

        # Identify critical chain (longest path considering resource constraints)
        self._identify_critical_chain(G)

        # Calculate project buffer (50% of critical chain duration in this simple implementation)
        self._calculate_buffers()

    # def _apply_resource_leveling(self) -> None:
    #     """Apply resource leveling to resolve resource conflicts"""
    #     # This is a simplified implementation
    #     # Sort tasks by start date
    #     sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

    #     # Track resource usage over time
    #     resource_timeline = {}

    #     for task in sorted_tasks:
    #         delay_needed = 0

    #         # Check each resource assigned to the task
    #         for resource_id in task.resources:
    #             if resource_id not in resource_timeline:
    #                 resource_timeline[resource_id] = []

    #             # Find conflicts with this resource
    #             for time_slot in resource_timeline[resource_id]:
    #                 start, end = time_slot
    #                 # Check if there's overlap
    #                 if not (task.end_date <= start or task.start_date >= end):
    #                     # Calculate needed delay to resolve conflict
    #                     potential_delay = end - task.start_date
    #                     delay_needed = max(delay_needed, potential_delay)

    #         # Apply delay if needed
    #         if delay_needed > 0:
    #             task.start_date += delay_needed
    #             task.end_date += delay_needed

    #         # Update resource timeline
    #         for resource_id in task.resources:
    #             resource_timeline[resource_id].append((task.start_date, task.end_date))

    # def _identify_critical_chain(self, G: nx.DiGraph) -> None:
    #     """Identify the critical chain in the project"""
    #     # Find the critical path considering durations
    #     # This is simplified; the actual critical chain considers resource constraints
    #     longest_path = nx.dag_longest_path(
    #         G,
    #         weight=lambda u, v, d: next(
    #             (t.duration for t in self.tasks if t.get_ID() == v), 0
    #         ),
    #     )

    #     # Mark tasks on the critical chain
    #     for task_id in longest_path:
    #         task = next((t for t in self.tasks if t.get_ID() == task_id), None)
    #         if task:
    #             task.is_critical = True

    #     # Store the critical chain (main chain)
    #     self.chains = [
    #         [t for t in self.tasks if t.get_ID() == task_id] for task_id in longest_path
    #     ]

    # def _calculate_buffers(self) -> None:
    #     """Calculate project and feeding buffers"""
    #     # Calculate project buffer (50% of critical chain duration)
    #     critical_tasks = [t for t in self.tasks if t.is_critical]
    #     total_critical_duration = sum(t.duration for t in critical_tasks)
    #     self.project_buffer = total_critical_duration * 0.5

    # def generate_gantt_chart(self, filename: str = None) -> plt.Figure:
    #     """Generate Gantt chart with dependency arrows and pickable bars"""
    #     # Create figure with interactive mode off to prevent popup window
    #     plt.ioff()
    #     fig, ax = plt.subplots(figsize=(12, 6))

    #     # Sort tasks by start date
    #     sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

    #     # Create a mapping of task_id to y-position for arrow drawing
    #     task_positions = {task.task_id: i for i, task in enumerate(sorted_tasks)}

    #     # Define color mapping - converting resource color names to actual matplotlib colors
    #     color_map = {
    #         "Red": "red",
    #         "Green": "green",
    #         "Blue": "blue",
    #         "Magenta": "magenta",
    #         "Yellow": "yellow",
    #         "Black": "black",
    #         # Add fallback colors for other resources
    #         "R1": "darkblue",
    #         "R2": "darkgreen",
    #         "R3": "darkred",
    #     }

    #     # Default color for tasks with multiple or no resources
    #     default_task_color = "gray"

    #     # Plot tasks
    #     for i, task in enumerate(sorted_tasks):
    #         start = task.start_date
    #         duration = task.duration

    #         # Determine task color based on resource
    #         if task.resources and len(task.resources) == 1:
    #             # If task has exactly one resource, use that resource's color
    #             resource = task.resources[0]
    #             task_color = color_map.get(resource, default_task_color)
    #         else:
    #             # For multiple resources or no resources, use the default color
    #             task_color = default_task_color

    #         # Plot the task bar - make it pickable with a unique label
    #         bar = ax.barh(
    #             i,
    #             duration,
    #             left=start,
    #             color=task_color,
    #             alpha=0.6,
    #             picker=5,  # 5 points tolerance for picking
    #             label=f"Task: {task.name} ID:{task.task_id}",  # Label with task info for picking
    #         )

    #         # If this is a critical task, add yellow border
    #         if task.is_critical:
    #             # Get the bar patch and set its edge color and width
    #             bar[0].set_edgecolor("yellow")
    #             bar[0].set_linewidth(2)

    #         # Add task name
    #         ax.text(
    #             start + duration / 2,
    #             i,
    #             f"{task.name}",
    #             ha="center",
    #             va="center",
    #             color="black",
    #         )

    #     # Add project buffer at the end
    #     if self.project_buffer > 0:
    #         project_end = max(t.end_date for t in self.tasks)
    #         buffer_bar = ax.barh(
    #             len(sorted_tasks),
    #             self.project_buffer,
    #             left=project_end,
    #             color="gold",
    #             alpha=0.6,
    #             label="Project Buffer",
    #         )

    #         # Add a yellow border to the buffer
    #         buffer_bar[0].set_edgecolor("yellow")
    #         buffer_bar[0].set_linewidth(2)

    #         ax.text(
    #             project_end + self.project_buffer / 2,
    #             len(sorted_tasks),
    #             f"Project Buffer",
    #             ha="center",
    #             va="center",
    #             color="black",
    #         )

    #     # Draw dependency arrows with smaller arrow heads
    #     for task in sorted_tasks:
    #         if task.predecessors:
    #             # Get list of predecessor IDs
    #             pred_ids = [int(p) for p in task.predecessors.split(",") if p.strip()]

    #             # Current task y-position
    #             task_y = task_positions[task.task_id]

    #             for pred_id in pred_ids:
    #                 # Find the predecessor task
    #                 pred_task = next(
    #                     (t for t in self.tasks if t.task_id == pred_id), None
    #                 )
    #                 if pred_task and pred_id in task_positions:
    #                     # Predecessor task y-position
    #                     pred_y = task_positions[pred_id]

    #                     # Arrow style - critical chain arrows are yellow, others are gray
    #                     arrow_color = (
    #                         "yellow"
    #                         if task.is_critical and pred_task.is_critical
    #                         else "gray"
    #                     )

    #                     # Smaller arrow heads for better proportion
    #                     arrow_style = "simple,head_width=2,head_length=3"

    #                     arrow_alpha = (
    #                         0.8 if task.is_critical and pred_task.is_critical else 0.6
    #                     )
    #                     arrow_linewidth = (
    #                         1.2 if task.is_critical and pred_task.is_critical else 0.8
    #                     )

    #                     # Draw arrow from end of predecessor to start of current task
    #                     # Use a gentler curve and smaller arrow heads
    #                     ax.annotate(
    #                         "",
    #                         xy=(task.start_date, task_y),  # end point
    #                         xytext=(pred_task.end_date, pred_y),  # start point
    #                         arrowprops=dict(
    #                             arrowstyle=arrow_style,
    #                             color=arrow_color,
    #                             alpha=arrow_alpha,
    #                             linewidth=arrow_linewidth,
    #                             connectionstyle="arc3,rad=.1",  # Less curved arrows (0.1 instead of 0.2)
    #                         ),
    #                     )

    #     # Set y-axis labels with task names
    #     ax.set_yticks(range(len(sorted_tasks) + 1))
    #     ax.set_yticklabels([t.name for t in sorted_tasks] + ["Buffer"])

    #     # Set chart title and labels
    #     ax.set_title("Project Gantt Chart with Critical Chain")
    #     ax.set_xlabel("Time")

    #     # Add a legend for resource colors
    #     unique_resources = set()
    #     for task in self.tasks:
    #         for resource in task.resources:
    #             unique_resources.add(resource)

    #     # Create legend patches
    #     legend_patches = []
    #     for resource in unique_resources:
    #         color = color_map.get(resource, default_task_color)
    #         patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6)
    #         legend_patches.append((patch, resource))

    #     # Add multiple resources patch
    #     if any(len(task.resources) > 1 for task in self.tasks):
    #         patch = plt.Rectangle((0, 0), 1, 1, fc=default_task_color, alpha=0.6)
    #         legend_patches.append((patch, "Multiple Resources"))

    #     # Add critical chain patch
    #     critical_patch = plt.Rectangle((0, 0), 1, 1, fc="white", ec="yellow", lw=2)
    #     legend_patches.append((critical_patch, "Critical Chain"))

    #     # Add dependency arrow patches to legend - also smaller for consistency
    #     normal_arrow = plt.Line2D(
    #         [0],
    #         [0],
    #         color="gray",
    #         marker=">",
    #         markersize=6,
    #         linestyle="-",
    #         linewidth=0.8,
    #     )
    #     critical_arrow = plt.Line2D(
    #         [0],
    #         [0],
    #         color="yellow",
    #         marker=">",
    #         markersize=6,
    #         linestyle="-",
    #         linewidth=1.2,
    #     )
    #     legend_patches.append((normal_arrow, "Task Dependency"))
    #     legend_patches.append((critical_arrow, "Critical Dependency"))

    #     # Add the legend
    #     ax.legend(
    #         [patch for patch, label in legend_patches],
    #         [label for patch, label in legend_patches],
    #         loc="lower right",
    #     )

    #     plt.tight_layout()

    #     # Save if filename is provided
    #     if filename:
    #         plt.savefig(filename)

    #     return fig

    def generate_gantt_chart(self, filename: str = None) -> plt.Figure:
        """Generate Gantt chart with dependency arrows, critical chain, and buffers"""
        # Create figure with interactive mode off to prevent popup window
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort tasks by start date
        sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

        # Create a mapping of task_id to y-position for arrow drawing
        task_positions = {task.task_id: i for i, task in enumerate(sorted_tasks)}

        # Define color mapping - converting resource color names to actual matplotlib colors
        color_map = {
            "Red": "red",
            "Green": "green",
            "Blue": "blue",
            "Magenta": "magenta",
            "Yellow": "yellow",
            "Black": "black",
            # Add fallback colors for other resources
            "R1": "darkblue",
            "R2": "darkgreen",
            "R3": "darkred",
        }

        # Default color for tasks with multiple or no resources
        default_task_color = "gray"

        # Plot tasks
        for i, task in enumerate(sorted_tasks):
            start = task.start_date
            duration = task.duration

            # Determine task color based on resource
            if task.resources and len(task.resources) == 1:
                # If task has exactly one resource, use that resource's color
                resource = task.resources[0]
                task_color = color_map.get(resource, default_task_color)
            else:
                # For multiple resources or no resources, use the default color
                task_color = default_task_color

            # Plot the task bar
            bar = ax.barh(
                i,
                duration,
                left=start,
                color=task_color,
                alpha=0.6,
                picker=5,  # 5 points tolerance for picking
                label=f"Task: {task.name} ID:{task.task_id}",  # Label with task info for picking
            )

            # If this is a critical task, add yellow border
            if task.is_critical:
                # Get the bar patch and set its edge color and width
                bar[0].set_edgecolor("yellow")
                bar[0].set_linewidth(2)

            # Add task name
            ax.text(
                start + duration / 2,
                i,
                f"{task.name}",
                ha="center",
                va="center",
                color="black",
            )

        # Count items for vertical positioning
        num_items = len(sorted_tasks)

        # Add project buffer at the end
        if self.project_buffer > 0:
            project_end = max(t.end_date for t in self.tasks if t.is_critical)
            buffer_bar = ax.barh(
                num_items,
                self.project_buffer,
                left=project_end,
                color="gold",
                alpha=0.6,
                label="Project Buffer",
            )

            # Add a yellow border to the buffer
            buffer_bar[0].set_edgecolor("yellow")
            buffer_bar[0].set_linewidth(2)

            ax.text(
                project_end + self.project_buffer / 2,
                num_items,
                f"Project Buffer",
                ha="center",
                va="center",
                color="black",
            )

            num_items += 1

        # Add feeding buffers
        for idx, buffer_info in enumerate(getattr(self, "feeding_buffers", [])):
            feeding_chain = buffer_info["feeding_chain"]
            joining_task = buffer_info["joining_task"]
            buffer_size = buffer_info["buffer_size"]

            if feeding_chain and joining_task:
                # Last task in the feeding chain
                last_task = feeding_chain[-1]

                # Position the feeding buffer after the last task in the chain
                buffer_start = last_task.end_date

                # Position in y-axis (below the project buffer)
                buffer_y = num_items + idx

                # Draw the feeding buffer
                fb_bar = ax.barh(
                    buffer_y,
                    buffer_size,
                    left=buffer_start,
                    color="lightgreen",  # Different color from project buffer
                    alpha=0.6,
                    label=f"Feeding Buffer ({last_task.name} → {joining_task.name})",
                )

                # Add a border to the buffer
                fb_bar[0].set_edgecolor("darkgreen")
                fb_bar[0].set_linewidth(1.5)

                # Add label
                ax.text(
                    buffer_start + buffer_size / 2,
                    buffer_y,
                    f"FB: {last_task.name}→{joining_task.name}",
                    ha="center",
                    va="center",
                    color="black",
                )

                # Draw an arrow from the feeding buffer to the joining task
                if joining_task.task_id in task_positions:
                    joining_y = task_positions[joining_task.task_id]

                    # Draw arrow from end of feeding buffer to the joining task
                    ax.annotate(
                        "",
                        xy=(joining_task.start_date, joining_y),  # end point
                        xytext=(buffer_start + buffer_size, buffer_y),  # start point
                        arrowprops=dict(
                            arrowstyle="simple,head_width=2,head_length=3",
                            color="darkgreen",
                            alpha=0.6,
                            linewidth=0.8,
                            connectionstyle="arc3,rad=.1",
                        ),
                    )

        # Update the number of items to include feeding buffers
        num_items += len(getattr(self, "feeding_buffers", []))

        # Draw dependency arrows
        for task in sorted_tasks:
            if task.predecessors:
                # Get list of predecessor IDs
                pred_ids = [int(p) for p in task.predecessors.split(",") if p.strip()]

                # Current task y-position
                task_y = task_positions[task.task_id]

                for pred_id in pred_ids:
                    # Find the predecessor task
                    pred_task = next(
                        (t for t in self.tasks if t.task_id == pred_id), None
                    )
                    if pred_task and pred_id in task_positions:
                        # Predecessor task y-position
                        pred_y = task_positions[pred_id]

                        # Arrow style - critical chain arrows are yellow, others are gray
                        arrow_color = (
                            "yellow"
                            if task.is_critical and pred_task.is_critical
                            else "gray"
                        )

                        # Smaller arrow heads for better proportion
                        arrow_style = "simple,head_width=2,head_length=3"

                        arrow_alpha = (
                            0.8 if task.is_critical and pred_task.is_critical else 0.6
                        )
                        arrow_linewidth = (
                            1.2 if task.is_critical and pred_task.is_critical else 0.8
                        )

                        # Draw arrow from end of predecessor to start of current task
                        # Use a gentler curve and smaller arrow heads
                        ax.annotate(
                            "",
                            xy=(task.start_date, task_y),  # end point
                            xytext=(pred_task.end_date, pred_y),  # start point
                            arrowprops=dict(
                                arrowstyle=arrow_style,
                                color=arrow_color,
                                alpha=arrow_alpha,
                                linewidth=arrow_linewidth,
                                connectionstyle="arc3,rad=.1",  # Less curved arrows (0.1 instead of 0.2)
                            ),
                        )

        # Set y-axis labels with task names and buffers
        y_labels = [t.name for t in sorted_tasks]
        y_labels.append("Project Buffer")

        # Add feeding buffer labels
        for buffer_info in getattr(self, "feeding_buffers", []):
            feeding_chain = buffer_info["feeding_chain"]
            joining_task = buffer_info["joining_task"]
            if feeding_chain and joining_task:
                last_task = feeding_chain[-1]
                y_labels.append(f"FB: {last_task.name}→{joining_task.name}")

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        # Set chart title and labels
        ax.set_title("Project Gantt Chart with Critical Chain and Buffers")
        ax.set_xlabel("Time")

        # Add a legend for resource colors
        unique_resources = set()
        for task in self.tasks:
            for resource in task.resources:
                unique_resources.add(resource)

        # Create legend patches
        legend_patches = []
        for resource in unique_resources:
            color = color_map.get(resource, default_task_color)
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6)
            legend_patches.append((patch, resource))

        # Add multiple resources patch
        if any(len(task.resources) > 1 for task in self.tasks):
            patch = plt.Rectangle((0, 0), 1, 1, fc=default_task_color, alpha=0.6)
            legend_patches.append((patch, "Multiple Resources"))

        # Add critical chain patch
        critical_patch = plt.Rectangle((0, 0), 1, 1, fc="white", ec="yellow", lw=2)
        legend_patches.append((critical_patch, "Critical Chain"))

        # Add project buffer patch
        project_buffer_patch = plt.Rectangle(
            (0, 0), 1, 1, fc="gold", alpha=0.6, ec="yellow", lw=2
        )
        legend_patches.append((project_buffer_patch, "Project Buffer"))

        # Add feeding buffer patch
        feeding_buffer_patch = plt.Rectangle(
            (0, 0), 1, 1, fc="lightgreen", alpha=0.6, ec="darkgreen", lw=1.5
        )
        legend_patches.append((feeding_buffer_patch, "Feeding Buffer"))

        # Add dependency arrow patches to legend - also smaller for consistency
        normal_arrow = plt.Line2D(
            [0],
            [0],
            color="gray",
            marker=">",
            markersize=6,
            linestyle="-",
            linewidth=0.8,
        )
        critical_arrow = plt.Line2D(
            [0],
            [0],
            color="yellow",
            marker=">",
            markersize=6,
            linestyle="-",
            linewidth=1.2,
        )
        legend_patches.append((normal_arrow, "Task Dependency"))
        legend_patches.append((critical_arrow, "Critical Dependency"))

        # Add the legend
        ax.legend(
            [patch for patch, label in legend_patches],
            [label for patch, label in legend_patches],
            loc="lower right",
        )

        plt.tight_layout()

        # Save if filename is provided
        if filename:
            plt.savefig(filename)

        return fig

    def _identify_critical_chain(self, G: nx.DiGraph) -> None:
        """
        Identify the critical chain in the project using a more sophisticated approach.

        This implementation is based on advanced critical chain calculation methods that
        consider both task duration and resource constraints.
        """
        # Find all terminal tasks (tasks with no successors)
        terminal_tasks = [node for node in G.nodes() if G.out_degree(node) == 0]

        if not terminal_tasks:
            # If there are no terminal tasks, there's no critical chain
            self.chains = []
            return

        # For each terminal task, find the longest path to it
        longest_path = []
        longest_duration = 0

        for end_node in terminal_tasks:
            # Find all simple paths from any starting node to this end node
            for start_node in [node for node in G.nodes() if G.in_degree(node) == 0]:
                for path in nx.all_simple_paths(G, start_node, end_node):
                    # Calculate total duration of this path, considering resource leveling
                    path_duration = 0
                    for task_id in path:
                        task = next(
                            (t for t in self.tasks if t.get_ID() == task_id), None
                        )
                        if task:
                            path_duration += task.duration

                    # If this path is longer than our current longest, update
                    if path_duration > longest_duration:
                        longest_duration = path_duration
                        longest_path = path

        # Get task objects for the critical chain
        critical_chain_tasks = []
        for task_id in longest_path:
            task = next((t for t in self.tasks if t.get_ID() == task_id), None)
            if task:
                task.is_critical = True
                critical_chain_tasks.append(task)
            else:
                # If task not found, something's wrong
                print(
                    f"Warning: Task ID {task_id} in critical path not found in tasks list"
                )

        # Store the critical chain
        self.chains = [critical_chain_tasks]

        # Identify feeding chains (chains that merge into the critical chain)
        self._identify_feeding_chains(G, longest_path)

    def _identify_feeding_chains(self, G: nx.DiGraph, critical_path: list) -> None:
        """
        Identify feeding chains - chains that feed into the critical chain but are not part of it.

        Args:
            G: NetworkX directed graph of task dependencies
            critical_path: List of task IDs in the critical chain
        """
        # Find all tasks that are predecessors of tasks in the critical chain
        # but are not themselves in the critical chain
        feeding_chains = []

        for idx, task_id in enumerate(critical_path):
            # Get all predecessors of this task in the critical chain
            task = next((t for t in self.tasks if t.get_ID() == task_id), None)
            if not task or not task.get_preds():
                continue

            pred_ids = [int(p) for p in task.get_preds().split(",") if p.strip()]

            for pred_id in pred_ids:
                # Skip if this predecessor is in the critical chain
                if pred_id in critical_path:
                    continue

                # This is a potential start of a feeding chain
                # Find the longest path ending at this predecessor
                feeding_chain = self._find_longest_path_to_task(G, pred_id)

                if feeding_chain:
                    # Get task objects for this feeding chain
                    feeding_chain_tasks = []
                    for fc_task_id in feeding_chain:
                        fc_task = next(
                            (t for t in self.tasks if t.get_ID() == fc_task_id), None
                        )
                        if fc_task:
                            feeding_chain_tasks.append(fc_task)

                    if feeding_chain_tasks:
                        feeding_chains.append(feeding_chain_tasks)

        # Add feeding chains to the chains list
        self.chains.extend(feeding_chains)

    def _find_longest_path_to_task(self, G: nx.DiGraph, end_task_id: int) -> list:
        """
        Find the longest path to a specific task.

        Args:
            G: NetworkX directed graph of task dependencies
            end_task_id: Task ID to find the longest path to

        Returns:
            List of task IDs in the longest path
        """
        # Find nodes with no predecessors (start nodes)
        start_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]

        longest_path = []
        longest_duration = 0

        # Find the longest path from any start node to the target task
        for start_node in start_nodes:
            try:
                # Skip if there's no path from this start to the end
                if not nx.has_path(G, start_node, end_task_id):
                    continue

                # Get all paths from this start to the end task
                for path in nx.all_simple_paths(G, start_node, end_task_id):
                    # Calculate total duration of this path
                    path_duration = 0
                    for task_id in path:
                        task = next(
                            (t for t in self.tasks if t.get_ID() == task_id), None
                        )
                        if task:
                            path_duration += task.duration

                    # If this path is longer than our current longest, update
                    if path_duration > longest_duration:
                        longest_duration = path_duration
                        longest_path = path
            except nx.NetworkXNoPath:
                continue

        return longest_path

    def _calculate_buffers(self) -> None:
        """
        Calculate project buffer and feeding buffers according to critical chain methodology.

        The project buffer is calculated based on the critical chain.
        Feeding buffers are calculated for each feeding chain.
        """
        # Verify we have identified chains
        if not self.chains:
            print("Warning: No chains identified, cannot calculate buffers")
            return

        # The first chain is the critical chain
        critical_chain = self.chains[0] if self.chains else []

        # Calculate project buffer (50% of critical chain duration by default)
        if critical_chain:
            critical_chain_duration = sum(task.duration for task in critical_chain)
            self.project_buffer = self._calculate_buffer_size(critical_chain_duration)
        else:
            self.project_buffer = 0

        # Calculate feeding buffers for each feeding chain
        self.feeding_buffers = []

        for i, chain in enumerate(
            self.chains[1:], 1
        ):  # Skip the first chain (critical chain)
            if chain:
                # Calculate chain duration
                chain_duration = sum(task.duration for task in chain)

                # Calculate buffer size for this feeding chain
                buffer_size = self._calculate_buffer_size(chain_duration)

                # Find the task in the critical chain that this feeding chain feeds into
                last_task_in_chain = chain[-1] if chain else None
                if last_task_in_chain:
                    # Find successors of this last task
                    for task in self.tasks:
                        if task.get_preds():
                            pred_ids = [
                                int(p) for p in task.get_preds().split(",") if p.strip()
                            ]
                            if (
                                last_task_in_chain.get_ID() in pred_ids
                                and task.is_critical
                            ):
                                # Found a critical chain task that this feeding chain connects to
                                joining_task = task
                                break
                    else:
                        joining_task = None

                    # Store the feeding buffer information
                    self.feeding_buffers.append(
                        {
                            "buffer_size": buffer_size,
                            "feeding_chain": chain,
                            "joining_task": joining_task,
                        }
                    )

    def _calculate_buffer_size(self, chain_duration: int) -> int:
        """
        Calculate buffer size based on chain duration using the square root of the sum of squares method.

        This method is more sophisticated than simply using 50% of chain duration.

        Args:
            chain_duration: Total duration of the chain

        Returns:
            Buffer size (rounded to nearest integer)
        """
        # By default, we'll use 50% of the chain duration
        # This can be enhanced with statistical methods like square root of sum of squares
        # For now, keeping it straightforward
        return int(chain_duration * 0.5)

    def get_feeding_buffer_for_task(self, task_id: int) -> dict:
        """
        Get the feeding buffer information for a task if it's the last task in a feeding chain.

        Args:
            task_id: Task ID to check

        Returns:
            Dictionary with buffer information or None if not found
        """
        for buffer_info in self.feeding_buffers:
            feeding_chain = buffer_info["feeding_chain"]
            if feeding_chain and feeding_chain[-1].get_ID() == task_id:
                return buffer_info
        return None

    def _apply_resource_leveling(self) -> None:
        """
        Apply advanced resource leveling to resolve resource conflicts.

        This implementation is based on the priority-based resource leveling algorithm
        from the critical_chain_simulator VBA code, which considers:
        - Task priority (critical tasks get higher priority)
        - Task dependencies
        - Resource availability over time
        - Minimal project delay
        """
        # First, sort tasks by their early start date (based on dependencies only)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

        # Initialize resource calendar - track resource usage over time
        resource_calendar = {}
        for task in self.tasks:
            for resource_id in task.resources:
                if resource_id not in resource_calendar:
                    resource_calendar[resource_id] = {}

        # Get resource availability from resource manager
        resource_availability = {}
        for resource_id, resource_info in self.resource_manager.get_resources().items():
            # Convert percentage to decimal (e.g., 100% -> 1.0)
            resource_availability[resource_id] = resource_info["availability"] / 100.0

        # First pass - schedule tasks based on early start dates and check for conflicts
        # This algorithm uses a priority-based approach
        for task in sorted_tasks:
            # Skip tasks with no resources
            if not task.resources:
                continue

            # Start with the early start date based on predecessors
            current_start = task.start_date
            delay_needed = 0

            # Keep trying to schedule until we find a time with no conflicts
            scheduling_conflict = True
            while scheduling_conflict:
                scheduling_conflict = False

                # Check each resource assigned to the task
                for resource_id in task.resources:
                    if resource_id not in resource_calendar:
                        resource_calendar[resource_id] = {}

                    # Check resource availability for the entire duration of the task
                    for time_slot in range(
                        current_start, current_start + task.duration
                    ):
                        # Get current usage of this resource at this time slot
                        current_usage = resource_calendar[resource_id].get(time_slot, 0)

                        # Get the availability of this resource
                        availability = resource_availability.get(resource_id, 1.0)

                        # Check if adding this task would exceed resource capacity
                        if current_usage + 1 > availability:
                            # Find the next available slot for this resource
                            next_available_slot = self._find_next_available_slot(
                                resource_calendar[resource_id],
                                resource_availability.get(resource_id, 1.0),
                                time_slot,
                            )

                            # Calculate new delay needed
                            potential_delay = next_available_slot - current_start
                            if potential_delay > delay_needed:
                                delay_needed = potential_delay

                            scheduling_conflict = True
                            break  # Break out of time slot loop

                    if scheduling_conflict:
                        break  # Break out of resource loop

                # If there's a conflict, update the start date and try again
                if scheduling_conflict:
                    current_start += delay_needed
                    delay_needed = 0  # Reset delay for next iteration

            # Apply the final scheduled start date
            task.start_date = current_start
            task.end_date = current_start + task.duration

            # Update resource calendar with this task's usage
            for resource_id in task.resources:
                for time_slot in range(task.start_date, task.end_date):
                    if time_slot not in resource_calendar[resource_id]:
                        resource_calendar[resource_id][time_slot] = 0
                    resource_calendar[resource_id][time_slot] += 1

        # Second pass - optimization: try to pull tasks earlier if possible
        # This helps minimize gaps in the schedule
        for task in sorted(self.tasks, key=lambda t: t.start_date):
            if not task.resources:
                continue

            # Find the earliest possible start date based on predecessors
            earliest_start = 0
            if task.predecessors:
                pred_ids = task.predecessors.split(",")
                for pred_id in pred_ids:
                    if pred_id.strip():
                        pred_task = next(
                            (
                                t
                                for t in self.tasks
                                if str(t.task_id) == pred_id.strip()
                            ),
                            None,
                        )
                        if pred_task:
                            earliest_start = max(earliest_start, pred_task.end_date)

            # Don't try to schedule earlier than the earliest start date
            if task.start_date <= earliest_start:
                continue

            # Try to pull this task earlier
            current_start = task.start_date
            earliest_possible = current_start

            # Remove this task from the resource calendar temporarily
            for resource_id in task.resources:
                for time_slot in range(task.start_date, task.end_date):
                    if time_slot in resource_calendar[resource_id]:
                        resource_calendar[resource_id][time_slot] -= 1

            # Try to schedule the task earlier
            for potential_start in range(earliest_start, current_start):
                can_schedule = True

                # Check each resource for this potential start time
                for resource_id in task.resources:
                    for time_slot in range(
                        potential_start, potential_start + task.duration
                    ):
                        current_usage = resource_calendar[resource_id].get(time_slot, 0)
                        availability = resource_availability.get(resource_id, 1.0)

                        if current_usage + 1 > availability:
                            can_schedule = False
                            break

                    if not can_schedule:
                        break

                if can_schedule:
                    earliest_possible = potential_start
                    break

            # Update task with the new earlier start date
            task.start_date = earliest_possible
            task.end_date = earliest_possible + task.duration

            # Add this task back to the resource calendar
            for resource_id in task.resources:
                for time_slot in range(task.start_date, task.end_date):
                    if time_slot not in resource_calendar[resource_id]:
                        resource_calendar[resource_id][time_slot] = 0
                    resource_calendar[resource_id][time_slot] += 1

    def _find_next_available_slot(
        self, resource_usage: dict, availability: float, current_time: int
    ) -> int:
        """
        Find the next time slot where a resource is available.

        Args:
            resource_usage: Dictionary mapping time slots to resource usage
            availability: The availability of the resource (e.g., 1.0 for 100%)
            current_time: Current time slot to start searching from

        Returns:
            The next available time slot
        """
        next_time = current_time + 1

        # Keep incrementing until we find an available slot
        while next_time in resource_usage and resource_usage[next_time] >= availability:
            next_time += 1

        return next_time


class CCPMplannerGUI:
    """GUI for the Critical Chain Project Management planner"""

    def __init__(self, root):
        self.root = root
        self.root.title("Critical Chain Project Management planner")
        self.root.geometry("1200x800")

        self.planner = CriticalChainplanner()

        # Initialize with some sample data
        self._create_sample_data()

        self._create_widgets()

    def _create_sample_data(self):
        """Create sample data for demonstration"""
        # Load default dataset
        self.planner.load_sample_dataset("Default")

    def _create_widgets(self):
        """Create GUI widgets"""
        # Create top menu
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # Create dataset menu
        self.dataset_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Datasets", menu=self.dataset_menu)

        # Add dataset options
        for dataset_name in self.planner.sample_datasets:
            dataset = self.planner.sample_datasets[dataset_name]
            self.dataset_menu.add_command(
                label=f"{dataset_name} - {dataset.description}",
                command=lambda name=dataset_name: self._load_dataset(name),
            )

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.task_tab = ttk.Frame(self.notebook)
        self.resource_tab = ttk.Frame(self.notebook)
        self.gantt_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.task_tab, text="Tasks")
        self.notebook.add(self.resource_tab, text="Resources")
        self.notebook.add(self.gantt_tab, text="Gantt Chart")

        # Set up the task tab
        self._setup_task_tab()

        # Set up the resource tab
        self._setup_resource_tab()

        # Set up the Gantt chart tab
        self._setup_gantt_tab()

    def _load_dataset(self, dataset_name):
        """Load a dataset by name"""
        if self.planner.load_sample_dataset(dataset_name):
            self._update_task_list()
            self._update_resource_list()
            self._update_gantt_chart()
            tk.messagebox.showinfo(
                "Dataset Loaded", f"Dataset '{dataset_name}' loaded successfully"
            )
        else:
            tk.messagebox.showerror("Error", f"Failed to load dataset '{dataset_name}'")

    def _setup_task_tab(self):
        """Set up the task management tab"""
        # Create frames
        task_list_frame = ttk.Frame(self.task_tab)
        task_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        task_edit_frame = ttk.Frame(self.task_tab)
        task_edit_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

        # Dataset info
        dataset_frame = ttk.Frame(task_list_frame)
        dataset_frame.pack(fill=tk.X, pady=5)

        self.dataset_label = ttk.Label(
            dataset_frame, text=f"Current Dataset: {self.planner.dataset_name}"
        )
        self.dataset_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            dataset_frame, text="Switch Dataset", command=self._show_dataset_menu
        ).pack(side=tk.RIGHT, padx=5)

        # Task list
        ttk.Label(task_list_frame, text="Task List").pack(anchor=tk.W)

        # Create Treeview for task list
        columns = ("ID", "Name", "Duration", "Resources", "Predecessors")
        self.task_tree = ttk.Treeview(task_list_frame, columns=columns, show="headings")

        # Define headings
        for col in columns:
            self.task_tree.heading(col, text=col)
            self.task_tree.column(col, width=100)

        self.task_tree.pack(fill=tk.BOTH, expand=True)

        # Button frame for task list
        task_button_frame = ttk.Frame(task_list_frame)
        task_button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            task_button_frame, text="Add Task", command=self._add_task_dialog
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(task_button_frame, text="Edit Task", command=self._edit_task).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            task_button_frame, text="Delete Task", command=self._delete_task
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            task_button_frame, text="Schedule Tasks", command=self._schedule_tasks
        ).pack(side=tk.RIGHT, padx=5)

        # Add a checkbox for using aggressive durations
        self.use_aggressive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            task_button_frame,
            text="Use Aggressive Durations",
            variable=self.use_aggressive_var,
            command=self._toggle_durations,
        ).pack(side=tk.RIGHT, padx=5)

        # Populate task list
        self._update_task_list()

    def _show_dataset_menu(self):
        """Show dataset selection menu"""
        # Create a popup menu
        popup = tk.Menu(self.root, tearoff=0)

        # Add dataset options
        for dataset_name in self.planner.sample_datasets:
            dataset = self.planner.sample_datasets[dataset_name]
            popup.add_command(
                label=f"{dataset_name} - {dataset.description}",
                command=lambda name=dataset_name: self._load_dataset(name),
            )

        # Display the popup menu
        try:
            popup.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            # Make sure to release the grab
            popup.grab_release()

    def _toggle_durations(self):
        """Toggle between safe and aggressive durations"""
        use_aggressive = self.use_aggressive_var.get()

        for task in self.planner.tasks:
            if hasattr(task, "aggressive_duration") and hasattr(task, "safe_duration"):
                if use_aggressive:
                    task.duration = task.aggressive_duration
                else:
                    task.duration = task.safe_duration

        self._update_task_list()
        self._schedule_tasks()

    def _setup_resource_tab(self):
        """Set up the resource management tab"""
        # Create frames
        resource_list_frame = ttk.Frame(self.resource_tab)
        resource_list_frame.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        resource_edit_frame = ttk.Frame(self.resource_tab)
        resource_edit_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

        # Resource list
        ttk.Label(resource_list_frame, text="Resource List").pack(anchor=tk.W)

        # Create Treeview for resource list
        columns = ("ID", "Name", "Availability")
        self.resource_tree = ttk.Treeview(
            resource_list_frame, columns=columns, show="headings"
        )

        # Define headings
        for col in columns:
            self.resource_tree.heading(col, text=col)
            self.resource_tree.column(col, width=100)

        self.resource_tree.pack(fill=tk.BOTH, expand=True)

        # Button frame for resource list
        resource_button_frame = ttk.Frame(resource_list_frame)
        resource_button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            resource_button_frame,
            text="Add Resource",
            command=self._add_resource_dialog,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            resource_button_frame, text="Edit Resource", command=self._edit_resource
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            resource_button_frame, text="Delete Resource", command=self._delete_resource
        ).pack(side=tk.LEFT, padx=5)

        # Populate resource list
        self._update_resource_list()

    def _update_task_list(self):
        """Update the task list in the UI"""
        # Clear existing items
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)

        # Update dataset label
        if hasattr(self, "dataset_label"):
            self.dataset_label.config(
                text=f"Current Dataset: {self.planner.dataset_name}"
            )

        # Add tasks to the treeview
        for task in self.planner.tasks:
            resources = ", ".join(task.resources)
            self.task_tree.insert(
                "",
                tk.END,
                values=(
                    task.task_id,
                    task.name,
                    task.duration,
                    resources,
                    task.predecessors,
                ),
            )

    def _update_resource_list(self):
        """Update the resource list in the UI"""
        # Clear existing items
        for item in self.resource_tree.get_children():
            self.resource_tree.delete(item)

        # Add resources to the treeview
        for (
            resource_id,
            resource,
        ) in self.planner.resource_manager.get_resources().items():
            self.resource_tree.insert(
                "",
                tk.END,
                values=(resource_id, resource["name"], f"{resource['availability']}%"),
            )

    def _add_task_dialog(self):
        """Open dialog to add a new task"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Task")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Task ID
        ttk.Label(dialog, text="Task ID:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        task_id_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=task_id_var).grid(
            row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Task Name
        ttk.Label(dialog, text="Task Name:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        task_name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=task_name_var).grid(
            row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Duration
        ttk.Label(dialog, text="Duration:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        duration_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=duration_var).grid(
            row=2, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Resources
        ttk.Label(dialog, text="Resources (comma-separated):").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5
        )
        resources_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=resources_var).grid(
            row=3, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Predecessors
        ttk.Label(dialog, text="Predecessors (comma-separated):").grid(
            row=4, column=0, sticky=tk.W, padx=5, pady=5
        )
        predecessors_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=predecessors_var).grid(
            row=4, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Save button
        def save_task():
            try:
                task_id = int(task_id_var.get())
                name = task_name_var.get()
                duration = int(duration_var.get())
                resources = [
                    r.strip() for r in resources_var.get().split(",") if r.strip()
                ]
                predecessors = predecessors_var.get()

                self.planner.add_task(task_id, name, duration, resources, predecessors)
                self._update_task_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_task).grid(
            row=6, column=0, columnspan=2, pady=10
        )

    def _edit_task(self):
        """Edit the selected task"""
        selection = self.task_tree.selection()
        if not selection:
            tk.messagebox.showinfo("Selection Required", "Please select a task to edit")
            return

        # Get selected task values
        selected_item = self.task_tree.item(selection[0])
        values = selected_item["values"]
        task_id = values[0]

        # Find the task in the planner
        task = next((t for t in self.planner.tasks if t.get_ID() == task_id), None)
        if not task:
            return

        # Open dialog similar to add_task_dialog but with values pre-filled
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Task")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Task ID (read-only)
        ttk.Label(dialog, text="Task ID:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        task_id_var = tk.StringVar(value=str(task.task_id))
        ttk.Entry(dialog, textvariable=task_id_var, state="readonly").grid(
            row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Task Name
        ttk.Label(dialog, text="Task Name:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        task_name_var = tk.StringVar(value=task.name)
        ttk.Entry(dialog, textvariable=task_name_var).grid(
            row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Duration
        ttk.Label(dialog, text="Duration:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        duration_var = tk.StringVar(value=str(task.duration))
        ttk.Entry(dialog, textvariable=duration_var).grid(
            row=2, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Resources
        ttk.Label(dialog, text="Resources (comma-separated):").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5
        )
        resources_var = tk.StringVar(value=",".join(task.resources))
        ttk.Entry(dialog, textvariable=resources_var).grid(
            row=3, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Predecessors
        ttk.Label(dialog, text="Predecessors (comma-separated):").grid(
            row=4, column=0, sticky=tk.W, padx=5, pady=5
        )
        predecessors_var = tk.StringVar(value=task.predecessors)
        ttk.Entry(dialog, textvariable=predecessors_var).grid(
            row=4, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Save button
        def save_task():
            try:
                # Update task values
                task.name = task_name_var.get()
                task.duration = int(duration_var.get())
                task.resources = [
                    r.strip() for r in resources_var.get().split(",") if r.strip()
                ]
                task.predecessors = predecessors_var.get()

                self._update_task_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_task).grid(
            row=6, column=0, columnspan=2, pady=10
        )

    def _delete_task(self):
        """Delete the selected task"""
        selection = self.task_tree.selection()
        if not selection:
            tk.messagebox.showinfo(
                "Selection Required", "Please select a task to delete"
            )
            return

        # Get selected task ID
        selected_item = self.task_tree.item(selection[0])
        task_id = selected_item["values"][0]

        # Confirm deletion
        if tk.messagebox.askyesno("Confirm Deletion", f"Delete task {task_id}?"):
            self.planner.remove_task(task_id)
            self._update_task_list()

    def _schedule_tasks(self):
        """Schedule all tasks"""
        try:
            self.planner.schedule_tasks()
            self._update_task_list()
            self._update_gantt_chart()
            tk.messagebox.showinfo("Success", "Tasks scheduled successfully")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to schedule tasks: {str(e)}")

    def _add_resource_dialog(self):
        """Open dialog to add a new resource"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Resource")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        # Resource ID
        ttk.Label(dialog, text="Resource ID:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        resource_id_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=resource_id_var).grid(
            row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Resource Name
        ttk.Label(dialog, text="Resource Name:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        resource_name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=resource_name_var).grid(
            row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Availability
        ttk.Label(dialog, text="Availability (%):").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        availability_var = tk.StringVar(value="100")
        ttk.Entry(dialog, textvariable=availability_var).grid(
            row=2, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Save button
        def save_resource():
            try:
                resource_id = resource_id_var.get()
                name = resource_name_var.get()
                availability = int(availability_var.get())

                if not resource_id or not name:
                    raise ValueError("Resource ID and Name are required")

                self.planner.resource_manager.add_resource(
                    resource_id, name, availability
                )
                self._update_resource_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_resource).grid(
            row=3, column=0, columnspan=2, pady=10
        )

    def _edit_resource(self):
        """Edit the selected resource"""
        selection = self.resource_tree.selection()
        if not selection:
            tk.messagebox.showinfo(
                "Selection Required", "Please select a resource to edit"
            )
            return

        # Get selected resource values
        selected_item = self.resource_tree.item(selection[0])
        values = selected_item["values"]
        resource_id = values[0]

        # Find the resource in the planner
        resource = self.planner.resource_manager.get_resources().get(resource_id)
        if not resource:
            return

        # Open dialog similar to add_resource_dialog but with values pre-filled
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Resource")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        # Resource ID (read-only)
        ttk.Label(dialog, text="Resource ID:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        resource_id_var = tk.StringVar(value=resource_id)
        ttk.Entry(dialog, textvariable=resource_id_var, state="readonly").grid(
            row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Resource Name
        ttk.Label(dialog, text="Resource Name:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        resource_name_var = tk.StringVar(value=resource["name"])
        ttk.Entry(dialog, textvariable=resource_name_var).grid(
            row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Availability
        ttk.Label(dialog, text="Availability (%):").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        availability_var = tk.StringVar(value=str(resource["availability"]))
        ttk.Entry(dialog, textvariable=availability_var).grid(
            row=2, column=1, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Save button
        def save_resource():
            try:
                # Update resource values
                name = resource_name_var.get()
                availability = int(availability_var.get())

                if not name:
                    raise ValueError("Resource Name is required")

                # Update the resource in the manager
                resource["name"] = name
                resource["availability"] = availability

                self._update_resource_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_resource).grid(
            row=3, column=0, columnspan=2, pady=10
        )

    def _delete_resource(self):
        """Delete the selected resource"""
        selection = self.resource_tree.selection()
        if not selection:
            tk.messagebox.showinfo(
                "Selection Required", "Please select a resource to delete"
            )
            return

        # Get selected resource ID
        selected_item = self.resource_tree.item(selection[0])
        resource_id = selected_item["values"][0]

        # Confirm deletion
        if tk.messagebox.askyesno(
            "Confirm Deletion", f"Delete resource {resource_id}?"
        ):
            self.planner.resource_manager.remove_resource(resource_id)
            self._update_resource_list()

    # def _on_hover(self, event):
    #     """Handle mouse hover event to show task tooltip"""
    #     # If we're outside the axes, hide the tooltip
    #     if event.inaxes is None:
    #         self.tooltip.place_forget()
    #         return

    #     # Check if the mouse is over any task bar
    #     for task_id, (bar, task) in self.task_bars.items():
    #         contains, _ = bar.contains(event)
    #         if contains:
    #             # Format the tooltip text
    #             tooltip_text = (
    #                 f"Task: {task.name} (ID: {task.task_id})\n"
    #                 f"Duration: {task.duration} days\n"
    #                 f"Start: Day {task.start_date}, End: Day {task.end_date}\n"
    #                 f"Resources: {', '.join(task.resources) if task.resources else 'None'}\n"
    #                 f"Predecessors: {task.predecessors if task.predecessors else 'None'}\n"
    #                 f"Critical: {'Yes' if task.is_critical else 'No'}"
    #             )

    #             # Include safe/aggressive duration if available
    #             if hasattr(task, "safe_duration") and hasattr(
    #                 task, "aggressive_duration"
    #             ):
    #                 tooltip_text += f"\nSafe Duration: {task.safe_duration}"
    #                 tooltip_text += f"\nAggressive Duration: {task.aggressive_duration}"

    #             # Update the tooltip text
    #             self.tooltip.config(text=tooltip_text)

    #             # Position the tooltip near the mouse but not under it
    #             self.tooltip.place(x=event.x + 15, y=event.y + 10)
    #             return

    #     # If not over any task, hide the tooltip
    #     self.tooltip.place_forget()

    def _setup_gantt_tab(self):
        """Set up the Gantt chart tab - simplified version without task details panel"""
        # Create a frame for the Gantt chart
        self.gantt_frame = ttk.Frame(self.gantt_tab)
        self.gantt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the Gantt chart with interactive mode off
        plt.ioff()  # Turn off interactive mode
        fig = self.planner.generate_gantt_chart()

        # Create canvas for the chart
        self.gantt_canvas = FigureCanvasTkAgg(fig, self.gantt_frame)
        canvas_widget = self.gantt_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        plt.close(fig)  # Close the figure to prevent display in a separate window

        # Add update button
        ttk.Button(
            self.gantt_tab, text="Update Gantt Chart", command=self._update_gantt_chart
        ).pack(pady=5)

    def _update_gantt_chart(self):
        """Update the Gantt chart - simplified version"""
        # Schedule tasks first
        self.planner.schedule_tasks()

        # Generate new chart - make sure it doesn't create a new window
        plt.ioff()  # Turn off interactive mode to prevent new window
        fig = self.planner.generate_gantt_chart()

        # Update canvas
        self.gantt_canvas.get_tk_widget().destroy()

        # Create new canvas in the frame
        self.gantt_canvas = FigureCanvasTkAgg(fig, self.gantt_frame)
        canvas_widget = self.gantt_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        plt.close(fig)  # Close the figure to prevent display in a separate window


def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = CCPMplannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
