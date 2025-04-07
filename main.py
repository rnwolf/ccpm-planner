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
        completion: float = 0.0,
    ):
        self.task_id = task_id
        self.name = name
        self.duration = duration
        self.resources = resources if resources else []
        self.predecessors = predecessors
        self.completion = completion
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

    def load_into_simulator(self, simulator):
        """Load this dataset into the provided simulator"""
        # Add resources
        for resource_id, availability in self.resources.items():
            simulator.resource_manager.add_resource(
                resource_id, resource_id, availability * 100
            )

        # Add tasks
        task_id = 1
        for task_data in self.tasks:
            if len(task_data) >= 5 and isinstance(
                task_data[0], str
            ):  # Larry Leech format with string names
                name, safe_duration, aggressive_duration, dependencies, resource = (
                    task_data
                )
                # Convert dependencies from task names to task IDs
                pred_ids = []
                for dep_name in dependencies:
                    for existing_task in simulator.tasks:
                        if existing_task.name == dep_name:
                            pred_ids.append(str(existing_task.task_id))
                            break

                # Create the task
                task = simulator.add_task(
                    task_id,
                    name,
                    int(safe_duration),  # Ensure safe duration is an integer
                    [resource] if resource else [],
                    ",".join(pred_ids),
                )

                # Set specific durations
                task.safe_duration = int(safe_duration)
                task.aggressive_duration = int(aggressive_duration)
            else:  # Standard format
                # Unpack task data
                if len(task_data) == 6:
                    task_id_val, name, duration, resources, predecessors, completion = (
                        task_data
                    )
                else:
                    # Default values for missing parameters
                    task_id_val, name, duration = (
                        task_data[0],
                        task_data[1],
                        task_data[2],
                    )
                    resources = task_data[3] if len(task_data) > 3 else []
                    predecessors = task_data[4] if len(task_data) > 4 else ""
                    completion = task_data[5] if len(task_data) > 5 else 0.0

                # Create the task
                simulator.add_task(
                    int(task_id_val),
                    name,
                    int(duration),
                    resources,
                    predecessors,
                    float(completion),
                )

                # Update task_id for next loop iteration
                task_id = int(task_id_val) + 1
                continue

            task_id += 1

        # Schedule tasks
        simulator.schedule_tasks()

        return simulator


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


class CriticalChainSimulator:
    """Main class for the Critical Chain Project Management simulator"""

    def __init__(self):
        self.tasks = []
        self.resource_manager = ResourceManager()
        self.chains = []  # Collections of tasks forming chains
        self.project_buffer = 0
        self.dataset_name = "Default"
        self.sample_datasets = self._create_sample_datasets()

    def _create_sample_datasets(self):
        """Create sample datasets"""
        datasets = {}

        # Default dataset
        default_resources = {"R1": 1, "R2": 1, "R3": 1}
        default_tasks = [
            (1, "Project Planning", 5, ["R1"], ""),
            (2, "Design", 10, ["R1", "R2"], "1"),
            (3, "Development Phase 1", 15, ["R2"], "2"),
            (4, "Development Phase 2", 12, ["R1"], "2"),
            (5, "Testing", 8, ["R3"], "3,4"),
            (6, "Documentation", 7, ["R1"], "5"),
            (7, "Deployment", 4, ["R1", "R2", "R3"], "5,6"),
        ]
        datasets["Default"] = SampleDataset(
            "Default",
            "Default sample project with 7 tasks",
            default_resources,
            default_tasks,
        )

        # Small example from Larry Leech's book
        small_resources = {"Red": 1, "Green": 1, "Magenta": 1, "Blue": 1}
        small_tasks = [
            ("T1.1", 30, 15, [], "Red"),
            ("T1.2", 20, 10, ["T1.1"], "Green"),
            ("T3", 30, 15, ["T1.2", "T2.2"], "Magenta"),
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
            self.sample_datasets[dataset_name].load_into_simulator(self)
            return True
        return False

    def add_task(
        self,
        task_id: int,
        name: str,
        duration: int,
        resources: List[str] = None,
        predecessors: str = "",
        completion: float = 0.0,
    ) -> Task:
        """Add a task to the project"""
        task = Task(task_id, name, duration, resources, predecessors, completion)
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

    def _apply_resource_leveling(self) -> None:
        """Apply resource leveling to resolve resource conflicts"""
        # This is a simplified implementation
        # Sort tasks by start date
        sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

        # Track resource usage over time
        resource_timeline = {}

        for task in sorted_tasks:
            delay_needed = 0

            # Check each resource assigned to the task
            for resource_id in task.resources:
                if resource_id not in resource_timeline:
                    resource_timeline[resource_id] = []

                # Find conflicts with this resource
                for time_slot in resource_timeline[resource_id]:
                    start, end = time_slot
                    # Check if there's overlap
                    if not (task.end_date <= start or task.start_date >= end):
                        # Calculate needed delay to resolve conflict
                        potential_delay = end - task.start_date
                        delay_needed = max(delay_needed, potential_delay)

            # Apply delay if needed
            if delay_needed > 0:
                task.start_date += delay_needed
                task.end_date += delay_needed

            # Update resource timeline
            for resource_id in task.resources:
                resource_timeline[resource_id].append((task.start_date, task.end_date))

    def _identify_critical_chain(self, G: nx.DiGraph) -> None:
        """Identify the critical chain in the project"""
        # Find the critical path considering durations
        # This is simplified; the actual critical chain considers resource constraints
        longest_path = nx.dag_longest_path(
            G,
            weight=lambda u, v, d: next(
                (t.duration for t in self.tasks if t.get_ID() == v), 0
            ),
        )

        # Mark tasks on the critical chain
        for task_id in longest_path:
            task = next((t for t in self.tasks if t.get_ID() == task_id), None)
            if task:
                task.is_critical = True

        # Store the critical chain (main chain)
        self.chains = [
            [t for t in self.tasks if t.get_ID() == task_id] for task_id in longest_path
        ]

    def _calculate_buffers(self) -> None:
        """Calculate project and feeding buffers"""
        # Calculate project buffer (50% of critical chain duration)
        critical_tasks = [t for t in self.tasks if t.is_critical]
        total_critical_duration = sum(t.duration for t in critical_tasks)
        self.project_buffer = total_critical_duration * 0.5

    def generate_gantt_chart(self, filename: str = None) -> plt.Figure:
        """Generate Gantt chart"""
        # Create figure with interactive mode off to prevent popup window
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort tasks by start date
        sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

        # Plot tasks
        for i, task in enumerate(sorted_tasks):
            start = task.start_date
            duration = task.duration

            # Different color for critical tasks
            color = "red" if task.is_critical else "blue"

            # Plot the task bar
            ax.barh(i, duration, left=start, color=color, alpha=0.6)

            # Add task name and completion percentage
            ax.text(
                start + duration / 2,
                i,
                f"{task.name} ({task.completion:.0%})",
                ha="center",
                va="center",
                color="black",
            )

        # Add project buffer at the end
        if self.project_buffer > 0:
            project_end = max(t.end_date for t in self.tasks)
            ax.barh(
                len(sorted_tasks),
                self.project_buffer,
                left=project_end,
                color="green",
                alpha=0.6,
            )
            ax.text(
                project_end + self.project_buffer / 2,
                len(sorted_tasks),
                f"Project Buffer",
                ha="center",
                va="center",
                color="black",
            )

        # Set y-axis labels with task names
        ax.set_yticks(range(len(sorted_tasks) + 1))
        ax.set_yticklabels([t.name for t in sorted_tasks] + ["Buffer"])

        # Set chart title and labels
        ax.set_title("Project Gantt Chart with Critical Chain")
        ax.set_xlabel("Time")

        plt.tight_layout()

        # Save if filename is provided
        if filename:
            plt.savefig(filename)

        return fig

    def generate_fever_chart(self, filename: str = None) -> plt.Figure:
        """Generate fever chart for project buffer consumption"""
        # Create figure with interactive mode off to prevent popup window
        plt.ioff()
        fig, ax = plt.subplots(figsize=(10, 6))

        # Example data points (in a real app, these would come from project tracking)
        completion_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        buffer_consumed = [0, 5, 12, 15, 20, 25, 28, 35, 40, 45, 50]

        # Plot the fever chart
        ax.plot(completion_percentages, buffer_consumed, "o-", color="blue")

        # Add reference lines
        # Green zone (safe)
        ax.fill_between([0, 33, 67, 100], [0, 33, 67, 100], color="green", alpha=0.2)

        # Yellow zone (warning)
        ax.fill_between(
            [0, 33, 67, 100],
            [33, 67, 100, 100],
            [0, 33, 67, 100],
            color="yellow",
            alpha=0.2,
        )

        # Red zone (danger)
        ax.fill_between(
            [0, 33, 67, 100],
            [100, 100, 100, 100],
            [33, 67, 100, 100],
            color="red",
            alpha=0.2,
        )

        # Set chart title and labels
        ax.set_title("Project Buffer Consumption Fever Chart")
        ax.set_xlabel("Project Completion (%)")
        ax.set_ylabel("Buffer Consumption (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True)

        plt.tight_layout()

        # Save if filename is provided
        if filename:
            plt.savefig(filename)

        return fig


class CCPMSimulatorGUI:
    """GUI for the Critical Chain Project Management simulator"""

    def __init__(self, root):
        self.root = root
        self.root.title("Critical Chain Project Management Simulator")
        self.root.geometry("1200x800")

        self.simulator = CriticalChainSimulator()

        # Initialize with some sample data
        self._create_sample_data()

        self._create_widgets()

    def _create_sample_data(self):
        """Create sample data for demonstration"""
        # Load default dataset
        self.simulator.load_sample_dataset("Default")

    def _create_widgets(self):
        """Create GUI widgets"""
        # Create top menu
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # Create dataset menu
        self.dataset_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Datasets", menu=self.dataset_menu)

        # Add dataset options
        for dataset_name in self.simulator.sample_datasets:
            dataset = self.simulator.sample_datasets[dataset_name]
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
        self.fever_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.task_tab, text="Tasks")
        self.notebook.add(self.resource_tab, text="Resources")
        self.notebook.add(self.gantt_tab, text="Gantt Chart")
        self.notebook.add(self.fever_tab, text="Fever Chart")

        # Set up the task tab
        self._setup_task_tab()

        # Set up the resource tab
        self._setup_resource_tab()

        # Set up the Gantt chart tab
        self._setup_gantt_tab()

        # Set up the fever chart tab
        self._setup_fever_tab()

    def _load_dataset(self, dataset_name):
        """Load a dataset by name"""
        if self.simulator.load_sample_dataset(dataset_name):
            self._update_task_list()
            self._update_resource_list()
            self._update_gantt_chart()
            self._update_fever_chart()
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
            dataset_frame, text=f"Current Dataset: {self.simulator.dataset_name}"
        )
        self.dataset_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            dataset_frame, text="Switch Dataset", command=self._show_dataset_menu
        ).pack(side=tk.RIGHT, padx=5)

        # Task list
        ttk.Label(task_list_frame, text="Task List").pack(anchor=tk.W)

        # Create Treeview for task list
        columns = ("ID", "Name", "Duration", "Resources", "Predecessors", "Completion")
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
        for dataset_name in self.simulator.sample_datasets:
            dataset = self.simulator.sample_datasets[dataset_name]
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

        for task in self.simulator.tasks:
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

    def _setup_gantt_tab(self):
        """Set up the Gantt chart tab"""
        # Create a frame for the Gantt chart
        self.gantt_frame = ttk.Frame(self.gantt_tab)
        self.gantt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the Gantt chart with interactive mode off
        plt.ioff()  # Turn off interactive mode
        fig = self.simulator.generate_gantt_chart()

        # Create canvas for the chart
        self.gantt_canvas = FigureCanvasTkAgg(fig, self.gantt_frame)
        self.gantt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)  # Close the figure to prevent display in a separate window

        # Add update button
        ttk.Button(
            self.gantt_tab, text="Update Gantt Chart", command=self._update_gantt_chart
        ).pack(pady=5)

    def _setup_fever_tab(self):
        """Set up the fever chart tab"""
        # Create a frame for the fever chart
        self.fever_frame = ttk.Frame(self.fever_tab)
        self.fever_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the fever chart with interactive mode off
        plt.ioff()  # Turn off interactive mode
        fig = self.simulator.generate_fever_chart()

        # Create canvas for the chart
        self.fever_canvas = FigureCanvasTkAgg(fig, self.fever_frame)
        self.fever_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)  # Close the figure to prevent display in a separate window

        # Add update button
        ttk.Button(
            self.fever_tab, text="Update Fever Chart", command=self._update_fever_chart
        ).pack(pady=5)

    def _update_task_list(self):
        """Update the task list in the UI"""
        # Clear existing items
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)

        # Update dataset label
        if hasattr(self, "dataset_label"):
            self.dataset_label.config(
                text=f"Current Dataset: {self.simulator.dataset_name}"
            )

        # Add tasks to the treeview
        for task in self.simulator.tasks:
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
                    f"{task.completion:.0%}",
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
        ) in self.simulator.resource_manager.get_resources().items():
            self.resource_tree.insert(
                "",
                tk.END,
                values=(resource_id, resource["name"], f"{resource['availability']}%"),
            )

    def _update_gantt_chart(self):
        """Update the Gantt chart"""
        # Schedule tasks first
        self.simulator.schedule_tasks()

        # Generate new chart - make sure it doesn't create a new window
        plt.ioff()  # Turn off interactive mode to prevent new window
        fig = self.simulator.generate_gantt_chart()

        # Update canvas
        self.gantt_canvas.get_tk_widget().destroy()

        # Create a frame for the Gantt chart if it doesn't exist
        if not hasattr(self, "gantt_frame") or not self.gantt_frame.winfo_exists():
            self.gantt_frame = ttk.Frame(self.gantt_tab)
            self.gantt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create new canvas in the frame
        self.gantt_canvas = FigureCanvasTkAgg(fig, self.gantt_frame)
        self.gantt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)  # Close the figure to prevent display in a separate window

    def _update_fever_chart(self):
        """Update the fever chart"""
        # Generate new chart - make sure it doesn't create a new window
        plt.ioff()  # Turn off interactive mode to prevent new window
        fig = self.simulator.generate_fever_chart()

        # Update canvas
        self.fever_canvas.get_tk_widget().destroy()

        # Create a frame for the fever chart if it doesn't exist
        if not hasattr(self, "fever_frame") or not self.fever_frame.winfo_exists():
            self.fever_frame = ttk.Frame(self.fever_tab)
            self.fever_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create new canvas in the frame
        self.fever_canvas = FigureCanvasTkAgg(fig, self.fever_frame)
        self.fever_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)  # Close the figure to prevent display in a separate window

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

        # Completion
        ttk.Label(dialog, text="Completion (%):").grid(
            row=5, column=0, sticky=tk.W, padx=5, pady=5
        )
        completion_var = tk.StringVar(value="0")
        ttk.Entry(dialog, textvariable=completion_var).grid(
            row=5, column=1, sticky=tk.W + tk.E, padx=5, pady=5
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
                completion = float(completion_var.get()) / 100

                self.simulator.add_task(
                    task_id, name, duration, resources, predecessors, completion
                )
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

        # Find the task in the simulator
        task = next((t for t in self.simulator.tasks if t.get_ID() == task_id), None)
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

        # Completion
        ttk.Label(dialog, text="Completion (%):").grid(
            row=5, column=0, sticky=tk.W, padx=5, pady=5
        )
        completion_var = tk.StringVar(value=str(int(task.completion * 100)))
        ttk.Entry(dialog, textvariable=completion_var).grid(
            row=5, column=1, sticky=tk.W + tk.E, padx=5, pady=5
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
                task.completion = float(completion_var.get()) / 100

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
            self.simulator.remove_task(task_id)
            self._update_task_list()

    def _schedule_tasks(self):
        """Schedule all tasks"""
        try:
            self.simulator.schedule_tasks()
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

                self.simulator.resource_manager.add_resource(
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

        # Find the resource in the simulator
        resource = self.simulator.resource_manager.get_resources().get(resource_id)
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
            self.simulator.resource_manager.remove_resource(resource_id)
            self._update_resource_list()


def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = CCPMSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
