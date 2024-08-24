import wandb

def init_wandb(
    wandb_entity,
    wandb_proj,
    wandb_config,
    overwrite,
):
    if overwrite:
        wandb.init(
            project = wandb_proj,
            entity = wandb_entity,
            config = wandb_config,
        )
        return None
        
    # first check whether there exists a run with the same configuration
    api = wandb.Api(timeout=300)
    runs = api.runs(f"{wandb_entity}/{wandb_proj}")
    find_existing_run = None
    for run in runs:
        run_config_list = {k: v for k,v in run.config.items() if not k.startswith('_')}
        this_run = True
        for key in wandb_config:
            if key == 'overwrite':
                continue
            if (not key in run_config_list) or (run_config_list[key] != wandb_config[key]): 
                this_run = False
                break
        if this_run: 
            if run.state != 'finished' or run.state!= 'running' or find_existing_run is not None:
                # remove crashed one or duplicated one
                if run.state == 'running':
                    print('Remove running job: ', run.name)
                elif run.state != 'finished':
                    print(f"Remove crashed run: {run.name}")
                if run.state == 'finished':
                    print(f"Remove duplicated run: {run.name}")
                run.delete()
            else:
                find_existing_run = run

                print("########"*3)
                print(f"Find existing run in wandb: {run.name}")
                print("########"*3)
        
    # initialize wandb
    if find_existing_run is None:
            
        wandb.init(
            project = wandb_proj,
            entity = wandb_entity,
            config = wandb_config,
        )
    else:
        print('Not overwrite, and the job has been done! Exit!')
        exit(0)
        
    return find_existing_run