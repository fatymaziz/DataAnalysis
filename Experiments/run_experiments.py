import subprocess


def run_experiment(script_name):
    """
    Run a Python script and wait for it to complete.
    Args:
        script_name (str): The name of the Python script to run.
    """
    result = subprocess.run(['python3', script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully ran {script_name}")
    else:
        print(f"Error running {script_name}: {result.stderr}")

if __name__ == "__main__":
    scripts = [
        'Experiment1.py',
        'Experiment2.py',
        'Experiment3.py',
        'Experiment4.py'
    ]

    for script in scripts:
        run_experiment(script)
