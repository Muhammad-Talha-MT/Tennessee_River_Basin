# Running the attribution experiments on Frontier (OLCF, project geo173)

Data root: `/lustre/orion/geo173/proj-shared/tva_data` (copied with `rsync`, see the transfer commands in the chat history /
`~/tva_rsync.sh`). Code: this repository, cloned next to it. Each run uses one GPU (GCD); a node has 8.

## 1. Code and environment (login node)
```bash
cd /lustre/orion/geo173/proj-shared/tva_data
git clone git@github.com:Muhammad-Talha-MT/Tennessee_River_Basin.git     # needs a GitHub key on Frontier; otherwise rsync the
                                                                           # repo from the local machine (without .git)
bash Tennessee_River_Basin/scripts/frontier/setup_env.sh                   # modules, venv, ROCm PyTorch, imports
```
`setup_env.sh` is untested on Frontier: module names and the PyTorch ROCm wheel index (`TORCH_INDEX`) may need changing.

## 2. Test in a short job (before anything long)
```bash
export TVA_ROOT=/lustre/orion/geo173/proj-shared/tva_data
cd $TVA_ROOT/Tennessee_River_Basin && module load PrgEnv-gnu cray-python rocm && source $TVA_ROOT/venv/bin/activate
salloc -A geo173 -N 1 -t 0:30:00 -q debug          # debug queue: check the current OLCF queue names and limits
srun -N1 -n1 --gpus-per-node=8 python scripts/frontier/check_gpu.py        # expect ALL OK
srun -N1 -n1 -c 8 --gpus-per-node=8 python -W ignore -m tva_flood.hydro.train --stage B \
  --out $TVA_ROOT/models/attr/_timing --init $TVA_ROOT/models/gefs_hydro_A/best.pt --loss nse-q --head lstm --delta-weight 0 \
  --fold 1 --updates 20 --accumulate 8 --warmup 1 --validate-every 20 --validation-limit 10 --workers 4
```
The last line prints `minutes` for 20 updates. The L40S needed about 3 min per 20 updates with 8 samples each (a full 3000-update run
takes about 7.7 h). If a Frontier GCD is more than about 2x slower, use more nodes.

## 3. Run (batch)
```bash
cd $TVA_ROOT/Tennessee_River_Basin
for i in 1 2 3 4; do sbatch --export=ALL,PHASE=phase1 scripts/frontier/run_phase.sbatch; done   # 4 nodes = 32 GPUs
```
- All copies share the job list: a run is claimed by one job, finished runs (`DONE`) are skipped.
- `SCRATCH_UPDATES` is 8000 here (the full length of A0), so A3 and A6 are not under-trained.
- Wall time is 6 h per job (`-t` in `run_phase.sbatch`); check the queue limits. A run cut off at the limit resumes
  from its last checkpoint (every 100 updates) when the next job starts: just `sbatch` again.
- Other phases: `PHASE=seeds` (A0 and A5 seeds 2 and 3), `PHASE=phase2` (blocked folds, coverage; run phase1 first), `PHASE=extra`.
  `phase0` (matched LSTM, leakage, conservation) stays on the local machine.

## 4. Monitor
```bash
squeue -u talham
ls $TVA_ROOT/models/attr/*/DONE | wc -l                       # finished runs and evaluations
tail -n 3 $TVA_ROOT/logs/attr/A5_s1_f2.log                     # a run's progress
```
Phase 1 needs 24 trainings + lumped stage A + 4 evaluations. If a job was lost without freeing its claims, they
are taken over automatically once Slurm no longer lists that job; to clear all claims: `rm -rf $TVA_ROOT/models/attr/*.claim`.

## 5. Bring the results back (run on the local machine; reports are made there)
```bash
ssh -fNM -S ~/.ssh/olcf.sock talham@dtn.olcf.ornl.gov true      # if the connection has expired (token)
rsync -a --info=progress2 -e "ssh -o ControlPath=$HOME/.ssh/olcf.sock" \
  --exclude 'checkpoint_*.pt' --exclude 'resume.pt' --exclude '*.claim' --exclude '_timing' \
  talham@dtn.olcf.ornl.gov:/lustre/orion/geo173/proj-shared/tva_data/models/attr/ /data/tva_data/models/attr/
cd /home/talhamuh/tva_flood/tva_flood_clean && bash scripts/attribution_pipeline.sh report
```
The evaluations run with `--no-reaches --rains gefs` on Frontier (no every-reach protocol, no realized-rain diagnostic).
