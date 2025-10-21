import torch.distributed as dist
import os
import datetime
import time
import random

def main():

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_PROCID"))
    current_device = local_rank

    torch.cuda.set_device(current_device)

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank), flush=True)

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=1.0), init_method=f"tcp://{os.environ.get('MASTER_ADDR')}:9437", world_size=int(os.environ.get('SLURM_NTA>

    random.seed(rank)

    s = random.randint(10, 60)

    print(f"[{datetime.datetime.now()}] Rank {rank} sleeping for {s} seconds", flush=True)

    time.sleep(s)

    print(f"[{datetime.datetime.now()}] Rank {rank} hit the barrier",flush=True)

    dist.barrier()

    print(f"[{datetime.datetime.now()}] Rank {rank} After Barrier", flush=True)

    dist.destroy_process_group()

    print("Exiting Successfully",flush=True)

if __name__ == "__main__":
    main()
