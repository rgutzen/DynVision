sbatch --time=06:00:00 --mem=2GB --wrap "sleep infinity"

# How to Connect VS Code to a Greene Compute Node
# https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code

# On Greene login node, submit a dummy Slurm job to sleep there, 1 CPU core and 1GB memory for 2 hours. This step will enable you to login to compute nodes you have jobs running there.
# sbatch --time=02:00:00 --mem=1GB --wrap "sleep infinity"

# After the job is running, please check the compute node this job is running there with squeue

# [sw77@log-1 tmp]$ squeue -u sw77
# JOBID    PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
# 23587213 cpu_gpu       wrap     sw77  R      18:29      1 gr070

# In this example, the job is running on gr070

# On your local computer, add this block to ~/.ssh/config file

# Host greene-compute
# HostName <Compute Node your Jupyter Notebook is using e.g. gr070>
# User <NetID>
# ProxyJump <NetID>@greene.hpc.nyu.edu
# StrictHostKeyChecking no
# UserKnownHostsFile /dev/null
# LogLevel ERROR

# Here the ProxyJump is <NetID>@greene, if you are in campus, please change to YourNYUNetID@greene.hpc.nyu.edu. Please be sure to be connected to the NYU VPN (vpn.nyu.edu).

# Set up ssh login to Greene with ssh key without password (see section above)

# Now from VSCode, you can directly connect to  greene-compute nodes with:

# ssh greene-compute