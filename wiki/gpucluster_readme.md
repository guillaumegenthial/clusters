# reference

look at http://sherlock.stanford.edu/mediawiki/index.php/SLURMSubmit

# how to connect to gpu cluster

`ssh -L 16006:127.0.0.1:6006  <userid>@icme-gpu.stanford.edu`

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

# submit a job with slurm

write a run.sh file 

```
#!/usr/bin/env bash

#SBATCH --partition=k80
# set the number of nodes
#SBATCH --nodes=1
# set the number of GPU cards to use per node 
#SBATCH --gres=gpu:1
# set output file
#SBATCH -o "$(date +'%Y%m%d_%H%M%S').out"
# set name of job 
python embeddings.py -t
```

and submit it with

```
sbatch run.sh
```

# list resources

```
sinfo
```

