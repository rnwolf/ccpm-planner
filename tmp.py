def _add_task_dialog(self):
        """Open dialog to add a new task"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Task")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Task ID
        ttk.Label(dialog, text="Task ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        task_id_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=task_id_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Task Name
        ttk.Label(dialog, text="Task Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        task_name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=task_name_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Duration
        ttk.Label(dialog, text="Duration:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        duration_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=duration_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Resources
        ttk.Label(dialog, text="Resources (comma-separated):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        resources_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=resources_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Predecessors
        ttk.Label(dialog, text="Predecessors (comma-separated):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        predecessors_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=predecessors_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Completion
        ttk.Label(dialog, text="Completion (%):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        completion_var = tk.StringVar(value="0")
        ttk.Entry(dialog, textvariable=completion_var).grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Save button
        def save_task():
            try:
                task_id = int(task_id_var.get())
                name = task_name_var.get()
                duration = int(duration_var.get())
                resources = [r.strip() for r in resources_var.get().split(",") if r.strip()]
                predecessors = predecessors_var.get()
                completion = float(completion_var.get()) / 100

                self.simulator.add_task(task_id, name, duration, resources, predecessors, completion)
                self._update_task_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_task).grid(row=6, column=0, columnspan=2, pady=10)

    def _edit_task(self):
        """Edit the selected task"""
        selection = self.task_tree.selection()
        if not selection:
            tk.messagebox.showinfo("Selection Required", "Please select a task to edit")
            return

        # Get selected task values
        selected_item = self.task_tree.item(selection[0])
        values = selected_item['values']
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
        ttk.Label(dialog, text="Task ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        task_id_var = tk.StringVar(value=str(task.task_id))
        ttk.Entry(dialog, textvariable=task_id_var, state="readonly").grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Task Name
        ttk.Label(dialog, text="Task Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        task_name_var = tk.StringVar(value=task.name)
        ttk.Entry(dialog, textvariable=task_name_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Duration
        ttk.Label(dialog, text="Duration:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        duration_var = tk.StringVar(value=str(task.duration))
        ttk.Entry(dialog, textvariable=duration_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Resources
        ttk.Label(dialog, text="Resources (comma-separated):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        resources_var = tk.StringVar(value=",".join(task.resources))
        ttk.Entry(dialog, textvariable=resources_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Predecessors
        ttk.Label(dialog, text="Predecessors (comma-separated):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        predecessors_var = tk.StringVar(value=task.predecessors)
        ttk.Entry(dialog, textvariable=predecessors_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Completion
        ttk.Label(dialog, text="Completion (%):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        completion_var = tk.StringVar(value=str(int(task.completion * 100)))
        ttk.Entry(dialog, textvariable=completion_var).grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Save button
        def save_task():
            try:
                # Update task values
                task.name = task_name_var.get()
                task.duration = int(duration_var.get())
                task.resources = [r.strip() for r in resources_var.get().split(",") if r.strip()]
                task.predecessors = predecessors_var.get()
                task.completion = float(completion_var.get()) / 100

                self._update_task_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_task).grid(row=6, column=0, columnspan=2, pady=10)

    def _delete_task(self):
        """Delete the selected task"""
        selection = self.task_tree.selection()
        if not selection:
            tk.messagebox.showinfo("Selection Required", "Please select a task to delete")
            return

        # Get selected task ID
        selected_item = self.task_tree.item(selection[0])
        task_id = selected_item['values'][0]

        # Confirm deletion
        if tk.messagebox.askyesno("Confirm Deletion", f"Delete task {task_id}?"):
            self.simulator.remove_task(task_id)
            self._update_task_list()

    def _add_resource_dialog(self):
        """Open dialog to add a new resource"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Resource")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        # Resource ID
        ttk.Label(dialog, text="Resource ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        resource_id_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=resource_id_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Resource Name
        ttk.Label(dialog, text="Resource Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        resource_name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=resource_name_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Availability
        ttk.Label(dialog, text="Availability (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        availability_var = tk.StringVar(value="100")
        ttk.Entry(dialog, textvariable=availability_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Save button
        def save_resource():
            try:
                resource_id = resource_id_var.get()
                name = resource_name_var.get()
                availability = int(availability_var.get())

                if not resource_id or not name:
                    raise ValueError("Resource ID and Name are required")

                self.simulator.resource_manager.add_resource(resource_id, name, availability)
                self._update_resource_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_resource).grid(row=3, column=0, columnspan=2, pady=10)

    def _edit_resource(self):
        """Edit the selected resource"""
        selection = self.resource_tree.selection()
        if not selection:
            tk.messagebox.showinfo("Selection Required", "Please select a resource to edit")
            return

        # Get selected resource values
        selected_item = self.resource_tree.item(selection[0])
        values = selected_item['values']
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
        ttk.Label(dialog, text="Resource ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        resource_id_var = tk.StringVar(value=resource_id)
        ttk.Entry(dialog, textvariable=resource_id_var, state="readonly").grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Resource Name
        ttk.Label(dialog, text="Resource Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        resource_name_var = tk.StringVar(value=resource['name'])
        ttk.Entry(dialog, textvariable=resource_name_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Availability
        ttk.Label(dialog, text="Availability (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        availability_var = tk.StringVar(value=str(resource['availability']))
        ttk.Entry(dialog, textvariable=availability_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Save button
        def save_resource():
            try:
                # Update resource values
                name = resource_name_var.get()
                availability = int(availability_var.get())

                if not name:
                    raise ValueError("Resource Name is required")

                # Update the resource in the manager
                resource['name'] = name
                resource['availability'] = availability

                self._update_resource_list()
                dialog.destroy()

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dialog, text="Save", command=save_resource).grid(row=3, column=0, columnspan=2, pady=10)

    def _delete_resource(self):
        """Delete the selected resource"""
        selection = self.resource_tree.selection()
        if not selection:
            tk.messagebox.showinfo("Selection Required", "Please select a resource to delete")
            return

        # Get selected resource ID
        selected_item = self.resource_tree.item(selection[0])
        resource_id = selected_item['values'][0]

        # Confirm deletion
        if tk.messagebox.askyesno("Confirm Deletion", f"Delete resource {resource_id}?"):
            self.simulator.resource_manager.remove_resource(resource_id)
            self._update_resource_list()

    def _load_dataset(self, dataset_name):
        """Load a dataset by name"""
        if self.simulator.load_sample_dataset(dataset_name):
            self._update_task_list()
            self._update_resource_list()
            self._update_gantt_chart()

            # Also update CCPM tab Gantt chart if it exists
            if hasattr(self, 'ccpm_gantt_canvas'):
                self._update_ccpm_gantt()

            self._update_fever_chart()
            tk.messagebox.showinfo("Dataset Loaded", f"Dataset '{dataset_name}' loaded successfully")
        else:
            tk.messagebox.showerror("Error", f"Failed to load dataset '{dataset_name}'")

    def _schedule_tasks(self):
        """Schedule all tasks"""
        try:
            self.simulator.schedule_tasks(ccpm_step=0)  # Full scheduling
            self._update_task_list()
            self._update_gantt_chart()

            # Also update CCPM tab Gantt chart if it exists
            if hasattr(self, 'ccpm_gantt_canvas'):
                self._update_ccpm_gantt()

            tk.messagebox.showinfo("Success", "Tasks scheduled successfully")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to schedule tasks: {str(e)}")


def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = CCPMSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
        """Edit the selected task"""
        selection = self.task_tree.selection()
        if not selection:
            tk.messagebox.showinfo("Selection Required", "Please select a task to edit")
            return

        # Get selected task values
        selected_item = self.task_tree.item(selection[0])
        values = selected_item['values']
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
        ttk.Label(dialog, text="Task ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        task_id_var = tk.StringVar(value=str(task.task_id))
        ttk.Entry(dialog, textvariable=task_id_var, state="readonly").grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Task Name
        ttk.Label(dialog, text="Task Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        task_name_var = tk.StringVar(value=task.name)
        ttk.Entry(dialog, textvariable=task_name_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Duration
        ttk.Label(dialog, text="Duration:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        duration_var = tk.StringVar(value=str(task.duration))
        ttk.Entry(dialog, textvariable=duration_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Resources
        ttk.Label(dialog, text="Resources (comma-separated):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        resources_var = tk.StringVar(value=",".join(task.resources))
        ttk.Entry(dialog, textvariable=resources_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Predecessors
        ttk.Label(dialog, text="Predecessors (comma-separated):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        predecessors_var = tk.StringVar(value=task.predecessors)
        ttk.Entry(dialog, textvariable=predecessors_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Completion
        ttk.Label(dialog, text="Completion (%):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        completion_var = tk.StringVar(value=str(int(task.completion * 100)))
        ttk.Entry(dialog, textvariable=completion_var).grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=5)# ccpm_simulator.py
# Main entry point file

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Classes
class Task:
    """Class to represent a task in the Critical Chain model"""

    def __init__(self, task_id: int, name: str, duration: int, resources: List[str] = None,
                 predecessors: str = "", completion: float = 0.0):
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
            self.aggressive_duration = int(int(duration) * 0.5)  # Aggressive estimate by default
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


class ResourceManager:
    """Class to manage resources"""

    def __init__(self):
        self.resources = {}

    def add_resource(self, resource_id: str, name: str, availability: int = 100):
        """Add a resource"""
        self.resources[resource_id] = {
            'name': name,
            'availability': availability,  # Percentage of availability
            'assignments': []
        }

    def remove_resource(self, resource_id: str):
        """Remove a resource"""
        if resource_id in self.resources:
            del self.resources[resource_id]

    def get_resources(self):
        """Get all resources"""
        return self.resources


class SampleDataset:
    """Class to hold sample dataset information"""

    def __init__(self, name: str, description: str, resources: Dict[str, int],
                 tasks: List[Tuple], expected_chain: List[str] = None):
        self.name = name
        self.description = description
        self.resources = resources  # Dict of resource_id: availability
        self.tasks = tasks  # List of task parameter tuples
        self.expected_chain = expected_chain  # Expected critical chain task names

    def load_into_simulator(self, simulator):
        """Load this dataset into the provided simulator"""
        # Add resources
        for resource_id, availability in self.resources.items():
            simulator.resource_manager.add_resource(resource_id, resource_id, availability * 100)

        # Add tasks
        task_id = 1
        for task_data in self.tasks:
            if len(task_data) >= 5 and isinstance(task_data[0], str):  # Larry Leech format with string names
                name, safe_duration, aggressive_duration, dependencies, resource = task_data
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
                    ",".join(pred_ids)
                )

                # Set specific durations
                task.safe_duration = int(safe_duration)
                task.aggressive_duration = int(aggressive_duration)
            else:  # Standard format
                # Unpack task data
                if len(task_data) == 6:
                    task_id_val, name, duration, resources, predecessors, completion = task_data
                else:
                    # Default values for missing parameters
                    task_id_val, name, duration = task_data[0], task_data[1], task_data[2]
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
                    float(completion)
                )

                # Update task_id for next loop iteration
                task_id = int(task_id_val) + 1
                continue

            task_id += 1

        # Schedule tasks
        simulator.schedule_tasks()

        return simulator


class CriticalChainSimulator:
    """Main class for the Critical Chain Project Management simulator"""

    def __init__(self):
        self.tasks = []
        self.resource_manager = ResourceManager()
        self.chains = []  # Collections of tasks forming chains
        self.critical_chain = []  # The critical chain tasks
        self.feeding_chains = []  # Non-critical chains that feed into the critical chain
        self.project_buffer = 0
        self.feeding_buffers = []  # List of feeding buffers
        self.dataset_name = "Default"
        self.sample_datasets = self._create_sample_datasets()
        self.current_step = 0  # Current step in the CCPM process

    def _create_sample_datasets(self):
        """Create sample datasets"""
        datasets = {}

        # Default dataset
        default_resources = {"Blue": 1, "Magenta": 1, "Green": 1}
        default_tasks = [
            (1, "Project Planning", 5, ["Blue"], ""),
            (2, "Design", 10, ["Blue", "Magenta"], "1"),
            (3, "Development Phase 1", 15, ["Magenta"], "2"),
            (4, "Development Phase 2", 12, ["Blue"], "2"),
            (5, "Testing", 8, ["Green"], "3,4"),
            (6, "Documentation", 7, ["Blue"], "5"),
            (7, "Deployment", 4, ["Blue", "Magenta", "Green"], "5,6"),
        ]
        datasets["Default"] = SampleDataset(
            "Default",
            "Default sample project with 7 tasks",
            default_resources,
            default_tasks,
        )

        # Small example from Larry Leech's book
        small_resources = {
            "Red": 1,
            "Green": 1,
            "Magenta": 1,
            "Blue": 1
        }
        small_tasks = [
            ("T1.1", 30, 15, [], "Red"),
            ("T1.2", 20, 10, ["T1.1"], "Green"),
            ("T3", 30, 15, ["T1.2", "T2.2"], "Magenta"),
            ("T2.1", 20, 10, [], "Blue"),
            ("T2.2", 10, 5, ["T2.1"], "Green")
        ]
        datasets["Larry_Small"] = SampleDataset(
            "Larry_Small",
            "Small example from Larry Leech's book on CCPM",
            small_resources,
            small_tasks,
            ["T1.1", "T1.2", "T2.2", "T3"]
        )

        # Large example from Larry Leech's book
        large_resources = {
            "Red": 1,
            "Green": 1,
            "Magenta": 1,
            "Blue": 1,
            "Black": 1
        }
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
            ("Done", 0, 0, ["C-6", "A-6"], "Black")
        ]
        datasets["Larry_Large"] = SampleDataset(
            "Larry_Large",
            "Large example from Larry Leech's book on CCPM",
            large_resources,
            large_tasks,
            ["A-1", "A-2", "A-3", "A-4", "B-4", "A-5", "A-6", "C-5", "C-6"]
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

    def add_task(self, task_id: int, name: str, duration: int, resources: List[str] = None,
                 predecessors: str = "", completion: float = 0.0) -> Task:
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
                preds = task.get_preds().split(',')

                # Check if the removed task is a predecessor
                if str(task_id) in preds:
                    # Remove the task ID from the predecessors string
                    new_preds = [p for p in preds if int(p) != task_id]
                    task.set_preds(','.join(new_preds))

    def remove_chains_first_task(self) -> None:
        """Remove the first task from each chain and update the chains"""
        for i, chain in enumerate(self.chains):
            if len(chain) > 0:
                # Remove the first task
                chain.pop(0)

                if len(chain) > 0:
                    # Get the predecessors of the new first task
                    preds = chain[0].get_preds().split(',') if chain[0].get_preds() else []

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

    def schedule_tasks(self, ccpm_step=0):
        """Schedule tasks according to predecessors and resource constraints

        Args:
            ccpm_step (int): The CCPM process step to execute
                0 = Full scheduling (all steps)
                1 = Identify critical chain only
                2 = Add project buffer only
                3 = Add feeding buffers
        """
        self.current_step = max(self.current_step, ccpm_step)

        # Reset task dates and buffer information
        for task in self.tasks:
            task.start_date = 0
            task.end_date = 0
            task.is_critical = False

        self.critical_chain = []
        self.feeding_chains = []
        self.feeding_buffers = []
        self.project_buffer = 0

        # Create a directed graph to represent task dependencies
        G = nx.DiGraph()

        # Add tasks as nodes
        for task in self.tasks:
            G.add_node(task.get_ID(), task=task)

        # Add edges for dependencies
        for task in self.tasks:
            if task.get_preds():
                preds = task.get_preds().split(',')
                for pred in preds:
                    if pred:  # Skip empty strings
                        G.add_edge(int(pred), task.get_ID())

        # Ensure the graph is acyclic (no circular dependencies)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Task dependencies contain cycles")

        # Get topological sort (order to process tasks)
        task_order = list(nx.topological_sort(G))

        # Find all end nodes (tasks with no successors)
        end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

        # Step 1: Schedule tasks based on predecessors (late finish network)
        for task_id in task_order:
            task = next((t for t in self.tasks if t.get_ID() == task_id), None)
            if task:
                if task.get_preds():
                    preds = task.get_preds().split(',')
                    # Find the latest end date among predecessors
                    latest_end = self.max_fin_preds_reel([p for p in preds if p])
                    task.start_date = latest_end

                task.end_date = task.start_date + task.duration

        # If ccpm_step is 0 or greater than 1, perform resource leveling
        if ccpm_step == 0 or ccpm_step >= 1:
            # Step 1c-e: Apply resource leveling (resolve resource conflicts)
            self._apply_resource_leveling()

            # Step 1f: Identify critical chain (longest path considering resource constraints)
            self._identify_critical_chain(G)

        # If ccpm_step is 0 or greater than 2, add project buffer
        if ccpm_step == 0 or ccpm_step >= 2:
            # Step 2b: Calculate project buffer (50% of critical chain duration in this implementation)
            self._calculate_project_buffer()

        # If ccpm_step is 0 or greater than 3, add feeding buffers
        if ccpm_step == 0 or ccpm_step >= 3:
            # Step 3a: Add feeding buffers
            self._calculate_feeding_buffers(G)

            # Step 3b: Resolve any new resource contentions
            if self.feeding_buffers:
                self._apply_resource_leveling()

    def _apply_resource_leveling(self):
        """Apply resource leveling to resolve resource conflicts (Step 1c-e)"""
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
        """Identify the critical chain in the project (Step 1f)"""
        # Find the critical path considering resource constraints
        # This is simplified; the actual critical chain considers both
        # dependency and resource constraints

        # Find all end nodes (tasks with no successors)
        end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

        # If there are multiple end nodes, find the one with the latest end date
        if end_nodes:
            latest_end_node = max(end_nodes, key=lambda n:
                                next((t.end_date for t in self.tasks if t.get_ID() == n), 0))
        else:
            # If no end nodes, use the node with the latest end date
            latest_end_node = max(G.nodes(), key=lambda n:
                                next((t.end_date for t in self.tasks if t.get_ID() == n), 0))

        # Find the critical path from any start node to the latest end node
        longest_path = None
        longest_duration = 0

        # Try all possible start nodes (nodes with no predecessors)
        start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

        for start_node in start_nodes:
            # Find all paths from start_node to latest_end_node
            for path in nx.all_simple_paths(G, start_node, latest_end_node):
                # Calculate path duration including both task duration and resource delays
                path_duration = 0
                path_end = 0

                for i, node in enumerate(path):
                    task = next((t for t in self.tasks if t.get_ID() == node), None)
                    if task:
                        # Add task duration
                        path_duration += task.duration
                        # Update path end time
                        path_end = task.end_date

                # If this path is longer than the current longest, update longest path
                if path_end > longest_duration:
                    longest_path = path
                    longest_duration = path_end

        # If we found a path, mark tasks on the critical chain
        if longest_path:
            self.critical_chain = []
            for task_id in longest_path:
                task = next((t for t in self.tasks if t.get_ID() == task_id), None)
                if task:
                    task.is_critical = True
                    self.critical_chain.append(task)

    def _calculate_project_buffer(self) -> None:
        """Calculate project buffer (Step 2b)"""
        # Calculate project buffer (50% of critical chain duration)
        if self.critical_chain:
            critical_duration = sum(task.duration for task in self.critical_chain)
            self.project_buffer = int(critical_duration * 0.5)

    def _calculate_feeding_buffers(self, G: nx.DiGraph) -> None:
        """Calculate feeding buffers for paths feeding into the critical chain (Step 3a)"""
        # Reset feeding chains and buffers
        self.feeding_chains = []
        self.feeding_buffers = []

        # If no critical chain, can't calculate feeding buffers
        if not self.critical_chain:
            return

        # Get IDs of critical chain tasks
        critical_ids = [task.get_ID() for task in self.critical_chain]

        # Find all paths that feed into the critical chain
        # First, identify all nodes that have a successor on the critical chain
        feeding_nodes = []
        for node in G.nodes():
            # Skip if node is on critical chain
            if node in critical_ids:
                continue

            # Check if node feeds directly into critical chain
            for successor in G.successors(node):
                if successor in critical_ids:
                    feeding_nodes.append(node)
                    break

        # For each feeding node, find all paths to it from start nodes
        start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

        for feeding_node in feeding_nodes:
            for start_node in start_nodes:
                # Find all paths from start_node to feeding_node
                for path in nx.all_simple_paths(G, start_node, feeding_node):
                    # Skip if any node in path is on critical chain (except possibly the last)
                    if any(node in critical_ids for node in path[:-1]):
                        continue

                    # Calculate path duration
                    path_duration = sum(next((t.duration for t in self.tasks if t.get_ID() == node), 0)
                                       for node in path)

                    # Create a feeding chain
                    feeding_chain = [next((t for t in self.tasks if t.get_ID() == node), None)
                                   for node in path]
                    feeding_chain = [t for t in feeding_chain if t is not None]

                    if feeding_chain:
                        self.feeding_chains.append(feeding_chain)

                        # Calculate buffer size (50% of feeding chain duration)
                        buffer_size = int(path_duration * 0.5)

                        # Create a buffer entry
                        buffer_entry = {
                            'id': len(self.feeding_buffers) + 1,
                            'size': buffer_size,
                            'feeding_chain': feeding_chain,
                            'insertion_point': feeding_node,  # Node where buffer connects to critical chain
                            'start_date': feeding_chain[-1].end_date,  # Buffer starts after last task in chain
                            'end_date': feeding_chain[-1].end_date + buffer_size  # Buffer ends after its duration
                        }

                        self.feeding_buffers.append(buffer_entry)

    def generate_gantt_chart(self, filename: str = None) -> plt.Figure:
        """Generate Gantt chart"""
        # Create figure with interactive mode off to prevent popup window
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort tasks by start date
        sorted_tasks = sorted(self.tasks, key=lambda t: t.start_date)

        # Define color mapping - converting resource color names to actual matplotlib colors
        color_map = {
            'Red': 'red',
            'Green': 'green',
            'Blue': 'blue',
            'Magenta': 'magenta',
            'Yellow': 'yellow',
            'Black': 'black',
            # Add fallback colors for other resources
            'R1': 'darkblue',
            'R2': 'darkgreen',
            'R3': 'darkred'
        }

        # Default color for tasks with multiple or no resources
        default_task_color = 'gray'

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
            bar = ax.barh(i, duration, left=start, color=task_color, alpha=0.6)

            # If this is a critical task, add yellow border
            if task.is_critical:
                # Get the bar patch and set its edge color and width
                bar[0].set_edgecolor('yellow')
                bar[0].set_linewidth(2)

            # Add task name and completion percentage
            ax.text(start + duration/2, i,
                   f"{task.name} ({task.completion:.0%})",
                   ha='center', va='center', color='black')

        # Get the current ylim
        y_min, y_max = ax.get_ylim()
        row_count = len(sorted_tasks)

        # Add project buffer at the end
        if self.project_buffer > 0:
            project_end = max(t.end_date for t in self.tasks if t.is_critical)
            buffer_bar = ax.barh(row_count, self.project_buffer, left=project_end,
                   color='gold', alpha=0.6)

            # Add a yellow border to the buffer
            buffer_bar[0].set_edgecolor('yellow')
            buffer_bar[0].set_linewidth(2)

            ax.text(project_end + self.project_buffer/2, row_count,
                   f"Project Buffer", ha='center', va='center', color='black')

            row_count += 1

        # Add feeding buffers if they exist
        for i, buffer in enumerate(self.feeding_buffers):
            buffer_row = row_count + i
            buffer_start = buffer['start_date']
            buffer_size = buffer['size']

            # Plot the feeding buffer
            buffer_bar = ax.barh(buffer_row, buffer_size, left=buffer_start,
                               color='lightgreen', alpha=0.6)

            # Add text label
            ax.text(buffer_start + buffer_size/2, buffer_row,
                   f"Feeding Buffer {buffer['id']}",
                   ha='center', va='center', color='black')

        # Adjust number of rows for feeding buffers
        row_count += len(self.feeding_buffers)

        # Set y-axis labels with task names and buffer names
        y_labels = [t.name for t in sorted_tasks]
        if self.project_buffer > 0:
            y_labels.append("Project Buffer")
        for buffer in self.feeding_buffers:
            chain_desc = " -> ".join(t.name for t in buffer['feeding_chain'][-2:])
            y_labels.append(f"FB {buffer['id']}: {chain_desc}")

        ax.set_yticks(range(row_count))
        ax.set_yticklabels(y_labels)

        # Set chart title and labels
        title = "Project Gantt Chart"
        if self.current_step == 1:
            title += " - Step 1: Critical Chain Identified"
        elif self.current_step == 2:
            title += " - Step 2: Project Buffer Added"
        elif self.current_step >= 3:
            title += " - Step 3: Feeding Buffers Added"

        ax.set_title(title)
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
        critical_patch = plt.Rectangle((0, 0), 1, 1, fc='white', ec='yellow', lw=2)
        legend_patches.append((critical_patch, "Critical Chain"))

        # Add project buffer patch
        if self.project_buffer > 0:
            buffer_patch = plt.Rectangle((0, 0), 1, 1, fc='gold', alpha=0.6)
            legend_patches.append((buffer_patch, "Project Buffer"))

        # Add feeding buffer patch
        if self.feeding_buffers:
            fb_patch = plt.Rectangle((0, 0), 1, 1, fc='lightgreen', alpha=0.6)
            legend_patches.append((fb_patch, "Feeding Buffer"))

        # Add the legend
        ax.legend([patch for patch, label in legend_patches],
                 [label for patch, label in legend_patches],
                 loc='lower right')

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
        ax.plot(completion_percentages, buffer_consumed, 'o-', color='blue')

        # Add reference lines
        # Green zone (safe)
        ax.fill_between([0, 33, 67, 100], [0, 33, 67, 100], color='green', alpha=0.2)

        # Yellow zone (warning)
        ax.fill_between([0, 33, 67, 100], [33, 67, 100, 100], [0, 33, 67, 100], color='yellow', alpha=0.2)

        # Red zone (danger)
        ax.fill_between([0, 33, 67, 100], [100, 100, 100, 100], [33, 67, 100, 100], color='red', alpha=0.2)

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