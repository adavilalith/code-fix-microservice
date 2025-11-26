import difflib

def generate_unified_diff(original_code: str, secure_code: str, fromfile: str = 'vulnerable.py', tofile: str = 'secure.py') -> str:
    """
    Generates a unified diff string between two versions of code.
    
    Args:
        original_code: The vulnerable code snippet (the 'before' version).
        secure_code: The remediated, secure code (the 'after' version).
        fromfile: The label for the original code file (e.g., 'vulnerable.py').
        tofile: The label for the secure code file (e.g., 'secure.py').
        
    Returns:
        A multiline string containing the unified diff.
    """
    
    # 1. Split the code into lines
    original_lines = original_code.splitlines(keepends=True)
    secure_lines = secure_code.splitlines(keepends=True)
    
    # 2. Generate the diff
    diff_generator = difflib.unified_diff(
        original_lines,
        secure_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm='' # Important: Prevents double newlines since we used keepends=True
    )
    
    # 3. Join the lines into a single string
    diff_string = "".join(diff_generator)
    
    return diff_string

# --- Example Usage ---

vulnerable_code = (
    "import os\n"
    "def execute_command(user_input):\n"
    "    # Vulnerable to shell injection\n"
    "    os.system(user_input)\n"
    "    return 'Command executed.'\n"
)

secure_code = (
    "import subprocess\n" # Change
    "def execute_command(user_input):\n"
    "    # Secure using subprocess.run\n" # Change
    "    subprocess.run([user_input], shell=False)\n" # Change
    "    return 'Command executed.'\n"
)

git_diff = generate_unified_diff(vulnerable_code, secure_code)

print(git_diff)