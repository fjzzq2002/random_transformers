import subprocess
import time
from pprint import pprint

def check_gpu_usage():
    # Function to check GPU usage using nvidia-smi command
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        pprint(output)
        gpu_usage = sum(map(int,output.decode('utf-8').strip().split()))
        pprint(gpu_usage)
        return gpu_usage
    except Exception as e:
        print("Error occurred while checking GPU usage:", e)
        return None

def main():
    # to start, wait for 2h
    time.sleep(7200)  # 7200 seconds = 2 hours
    not_in_use = 0
    while True:
        # Wait for 1 minute
        time.sleep(60)
        gpu_usage = check_gpu_usage()
        if gpu_usage is not None:
            print("GPU Usage: {}%".format(gpu_usage))
            
            # If GPU usage is below a certain threshold, break out of the loop
            if gpu_usage < 1:
                not_in_use += 1
                if not_in_use > 120:
                    # If GPU is not in use for 2 hours, exit the loop
                    print("GPU is not in use for 2 hours. Exiting loop.")
                    break
            else:
                not_in_use = 0
        else:
            print("Error occurred while checking GPU usage. Exiting loop.")
            break

if __name__ == "__main__":
    main()