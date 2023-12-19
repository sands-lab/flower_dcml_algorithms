import wandb

api = wandb.Api()
runs = api.runs("borisrado/test-project")

for run in runs:
    print(run.history())
    print(run.config)
