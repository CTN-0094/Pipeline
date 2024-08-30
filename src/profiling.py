import cProfile  # For profiling the code
import pstats  # For working with profiling statistics
import io  # For in-memory text streams
import os  # For file and directory operations
import time
from datetime import datetime

def profile_pipeline(main_func, label):
    # Create a Profile object to start profiling
    pr = cProfile.Profile()
    
    # Enable the profiler to start recording profiling data
    pr.enable()
    
    # Call the main function of your script that you want to profile
    main_func(42, label)
    
    # Disable the profiler to stop recording profiling data
    pr.disable()
    
    # Create an in-memory text stream to hold the profiling results
    s = io.StringIO()
    
    # Specify how to sort the profiling statistics (e.g., by cumulative time)
    sortby = 'cumulative'
    
    # Create a Stats object with the profiling data and sort it
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    
    # Print the profiling statistics to the in-memory text stream
    ps.print_stats()
    
    # Define the profiling log file path
    log_dir = "Profiling_Logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add timestamp to the file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    profile_log_file = os.path.join(log_dir, f"profiling_results_{label}_{timestamp}.txt")
    
    # Write the contents of the in-memory text stream (the profiling results) to the log file
    with open(profile_log_file, "w") as f:
        f.write(s.getvalue())
    
    print(f"Profiling results written to {profile_log_file}")



def simple_profile_pipeline(main_func, label):
    # Call the main function of your script that you want to profile
    start_time = time.time()  # Start the timer
    main_func(42, label)
    end_time = time.time()    # End the timer
    
    # Calculate the execution time
    execution_time = end_time - start_time
    
    # Define the profiling log file path
    log_dir = "Profiling_Logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add timestamp to the file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    profile_log_file = os.path.join(log_dir, f"prof_res_{label}_{timestamp}.txt")
    
    # Write the profiling results and the execution time to the log file
    with open(profile_log_file, "w") as f:
        f.write(f"\nExecution time for {label}: {execution_time:.4f} seconds\n")
    
    print(f"Profiling results and execution time written to {profile_log_file}")



def profileAllOutcomes(main_func):
    available_outcomes = [
        
        'Rd_kostenB_1993'
    ]
    for outcome in available_outcomes:
        profile_pipeline(main_func, outcome)
