import os
import sys
import time

task_id = int(sys.argv[1])
hostname = os.uname().nodename

print(f"Task {task_id:02d} running on {hostname}", flush=True)
time.sleep(2)
print(f"Task {task_id:02d} finished on {hostname}", flush=True)
