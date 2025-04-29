import os
import importlib.util
class TaskManager:
    def __init__(self, tasks_folder="./src/eval/tasks"):
        self.tasks_folder = tasks_folder
    def _load_function(self, task_name, function_name):
        """Loads a specific function from a task folder."""
        task_path = os.path.join(self.tasks_folder, task_name, f"{function_name}.py")
        if not os.path.isfile(task_path):
            raise FileNotFoundError(f"{task_path} does not exist!")
        spec = importlib.util.spec_from_file_location(function_name, task_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, function_name):
            raise AttributeError(f"{function_name} not found in {task_path}")
        return getattr(module, function_name)
    def load_task(self, task_name):
        """Loads all functions for a given task and binds them to the manager."""
        for func_name in ["process_label", "process_prediction", "eval_function","process_and_save_dataset","get_input_instruction","get_output_instruction"]:
            func = self._load_function(task_name, func_name)
            setattr(self, func_name, func)
    
