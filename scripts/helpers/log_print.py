import os
import sys

# Define the directory where the log file will be stored
log_directory = os.getcwd()

# Define the path for the log file
log_file_path = os.path.join(log_directory, "print_log.txt")

# Open the log file in append mode
log_file = open(log_file_path, "a")

def custom_print(*args, **kwargs):
    """
    Custom print function that writes to a log file instead of standard output.
    """
    print(*args, **kwargs, file=log_file)  # Redirect the print output to the log file
    log_file.flush()  # Ensure the log is written to the file immediately

# Override the built-in print function
built_in_print = print  # Keep a reference to the original print function, in case needed
print = custom_print

# Inform that the custom print is active
print("Custom print function is active. All print statements will be logged to:", log_file_path)

# From this point onward, all print statements will be logged to the specified log file
