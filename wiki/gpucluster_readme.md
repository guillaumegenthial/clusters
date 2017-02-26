# reference

look at http://sherlock.stanford.edu/mediawiki/index.php/SLURMSubmit

# how to connect to gpu cluster

`ssh -l <username>  icme-gpu.stanford.edu`

# list slurm jobs

`squeue`

get detail of a job

`scontrol show job 106`

cancel your job

`scancel 135`

hold your job

`scontrol hold 139`

release your job

`scontrol release 139`


# get an interactive node with one gpu

`srun --partition=k80 --gres=gpu:1 --pty bash`

# load tensorflow module

`module load tensorflow`
