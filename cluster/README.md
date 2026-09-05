# SciTech HPC Cluster Access

Reusable access guide for the HPC course.

## 1. Connect to the login node

Use the username and password assigned to you by SciTech. Do **not** share credentials.

### From outside the campus/laboratory network

```bash
ssh -J <username>@ssh.iesci.tech <username>@10.205.20.10
```

### From the campus/laboratory network

```bash
ssh <username>@10.205.20.10
```

The node at `10.205.20.10` is the cluster login node (`rust`). Use it to prepare files, compile software when appropriate, inspect Slurm, and submit jobs.

Do **not** SSH directly to compute nodes such as `haskell`, `julia`, or `fortran`. Ask Slurm for compute resources instead.

## 2. Optional SSH configuration

Add this to `~/.ssh/config` on your own computer:

```text
Host hpc
    HostName 10.205.20.10
    User <username>
    ProxyJump <username>@ssh.iesci.tech
```

Then connect from outside campus with:

```bash
ssh hpc
```

## 3. Check the cluster

After login:

```bash
hostname
whoami
sinfo
```

You should be on the login node and `sinfo` should show the available Slurm partitions.

Current teaching partitions include:

- `cpu` — CPU batch jobs
- `gpu` — GPU jobs
- `interactive` — short interactive work

Availability and limits can change; always use `sinfo` to see the current state.

## 4. Request compute resources through Slurm

### Short interactive CPU shell

```bash
srun -p interactive --pty bash
```

### One GPU interactively

```bash
srun -p gpu --gpus=1 --pty bash
```

Once Slurm starts the allocation, check where you are:

```bash
hostname
```

When finished, leave the compute allocation:

```bash
exit
```

Do not keep interactive allocations open when you are not using them.

## 5. Useful Slurm commands

```bash
sinfo                 # cluster/partition status
squeue -u $USER       # your queued/running jobs
scancel <jobid>       # cancel a job
```

For classroom exercises with scarce resources such as GPUs, prefer short batch jobs when instructed by the teacher.

## Mental model

```text
Your laptop
   |
   | SSH (outside campus: via ssh.iesci.tech)
   v
rust  = login node
   |
   | Slurm: srun / sbatch
   v
haskell / julia / fortran = compute nodes
```

The login node is the front door. Slurm decides where computation runs.
