#!/usr/bin/env python
#SBATCH --ntasks=50 # Number of cores requested
#SBATCH --exclusive
#SBATCH -t 96:00:00
#SBATCH -p general # Partition to submit to
#SBATCH --mem-per-cpu=4000 # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o log%j_makecat.out # Standard out goes to this file
#SBATCH -e log%j_makecat.err # Standard err goes to this filehostname
#SBATCH -J makecat
#SBATCH --mail-type=END
#SBATCH --mail-user=eschlafly@gmail.com

import os

pythonexec = "/n/home13/schlafly/.conda/envs/my_root/bin/python"
srun = "srun --exclusive -N1 -n1 --mem-per-cpu=4000"
parallel = "parallel --delay 2 -j $SLURM_NTASKS --colsep ' ' --joblog log/runtask.log --resume-failed"  #  --resume"
os.system('cat allimlist.txt | %s "%s %s $HOME/crowdsource/python/decam_proc.py --verbose --resume --outdir cat --outmodelfn mod/{4} {1} {2} {3} >>log/{1/.}.log 2>&1"' % (parallel, srun, pythonexec))

# can add --retries 2, but then need to add --clobber somewhere and figure
# out how to output to a new log, not the original one.
