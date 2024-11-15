import subprocess


def run_initial_script(script_path):
    try:
        # Use subprocess.Popen to run the script and stream the output in real-time
        process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print real-time output from the script
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")  # Print the output line by line
        for stderr_line in iter(process.stderr.readline, ""):
            print(stderr_line, end="")  # Print any error line by line

        process.stdout.close()
        process.stderr.close()
        process.wait()  # Wait for the process to complete

        if process.returncode == 0:
            print("Script executed successfully!")
            return "Execution successful"  # Return a success message or other appropriate result
        else:
            print("Script execution failed!")
            return "Execution failed"  # Return failure message or any error code
    except Exception as e:
        print(f"Error running the script: {e}")
        return str(e)
