import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import wandb
from datetime import datetime
from environment import WANDB_INFO

# Initialize wandb API
api = wandb.Api()

# Fetch all runs of the project
runs = api.runs(f"{WANDB_INFO['entity']}/{WANDB_INFO['project']}")

# # Define the cutoff time
# cutoff_time = datetime.strptime("2024-04-30 00:00:00", "%Y-%m-%d %H:%M:%S")

# # Loop through the runs and delete if they were created before the cutoff_time
# delete_run = 0
# for run in runs:
#     created_at = run.created_at
#     created_at = datetime.fromisoformat(created_at)
#     if created_at < cutoff_time and run.config['ssm_method'] == 'ours':
#         print(f"Run created at {created_at} has been deleted!")
#         delete_run += 1
#         run.delete()
        
# print(delete_run)

for run in runs:
    if not 'max_n_training_samples' in run.config:
        run.config['max_n_training_samples'] = 1000000000000000
        run.update()