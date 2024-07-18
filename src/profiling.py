import cProfile  # For profiling the code
import pstats  # For working with profiling statistics
import io  # For in-memory text streams
import os  # For file and directory operations

def profile_pipeline(main_func):
    # Create a Profile object to start profiling
    pr = cProfile.Profile()
    
    # Enable the profiler to start recording profiling data
    pr.enable()
    
    # Call the main function of your script that you want to profile
    main_func()
    
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
    
    profile_log_file = os.path.join(log_dir, "profiling_results.log")
    
    # Write the contents of the in-memory text stream (the profiling results) to the log file
    with open(profile_log_file, 'w') as f:
        f.write(s.getvalue())
    
    print(f"Profiling results written to {profile_log_file}")
