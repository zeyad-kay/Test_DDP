The python script initializes the process groups and sleeps for a random number of seconds for each worker to simulate some work being done. I used `dist.barrier()` to wait for all workers to finish.

If we look at the logs, the single node one runs perfectly fine. I am running in debug mode so pytorch logs lots of stuff but the print statements are printted with correct timing and the script exits successfully:  

```
...
25  From Rank: 0, ==> Initializing Process Group...
26  From Rank: 1, ==> Initializing Process Group...
27  [2025-10-20 20:30:45.850668] Rank 1 sleeping for 18 seconds
28  [2025-10-20 20:30:45.857824] Rank 0 sleeping for 34 seconds
29  [2025-10-20 20:31:03.850794] Rank 1 hit the barrier
...
33  [2025-10-20 20:31:19.857928] Rank 0 hit the barrier
...
125  ... [2025-10-20 20:31:20.612111] Rank 0 After Barrier
126  ... [2025-10-20 20:31:20.612113] Rank 1 After Barrier
127  Exiting Successfully
128  Exiting Successfully
...
```

The multi-node one initializes the processes, but once a process hits the barrier and tries to communicate with the other it crashes. When looking at the logs, it looks like the nodes cannot communicate and after a certain number of retries, it crashes.

```
...
44  From Rank: 0, ==> Initializing Process Group...
45  From Rank: 1, ==> Initializing Process Group...
46  [2025-10-20 20:37:02.365140] Rank 1 sleeping for 18 seconds
47  [2025-10-20 20:37:02.367676] Rank 0 sleeping for 34 seconds
48  [2025-10-20 20:37:20.365285] Rank 1 hit the barrier
...
66  g15:2039491:2039656 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (1/34) after sleep for 100 msec
67  g15:2039491:2039656 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (2/34) after sleep for 200 msec
68  g15:2039491:2039656 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (3/34) after sleep for 300 msec
...
 111  [rank1]: Traceback (most recent call last):
 112  [rank1]:   File "/home/zeyadk/Test_DDP/test_ddp.py", line 43, in <module>
 113  [rank1]:     main()
 114  [rank1]:   File "/home/zeyadk/Test_DDP/test_ddp.py", line 34, in main
 115  [rank1]:     dist.barrier()
 116  [rank1]:   File "/local/scratch/zeyadk.3082812.0/env/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
 117  [rank1]:     return func(*args, **kwargs)
 118  [rank1]:            ^^^^^^^^^^^^^^^^^^^^^
 119  [rank1]:   File "/local/scratch/zeyadk.3082812.0/env/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 4635, in barrier
 120  [rank1]:     work = group.barrier(opts=opts)
 121  [rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^
 122  [rank1]: torch.distributed.DistBackendError: NCCL error in: /tmp/build_wheels_tmp.6999/python-3.11/torch/torch/csrc/distributed/c10d/NCCLUtils.cpp:77, remote process exited or there was a network error, NCCL version 2.26.2
 123  [rank1]: ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
 124  [rank1]: Last error:
 125  [rank1]: socketPollConnect: connect returned Connection refused, exceeded error retry count (35)
 126  [rank1]:[W1020 20:38:36.173850072 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
 127  srun: error: g15: task 1: Exited with exit code 1
 128  srun: Terminating StepId=3082812.1
 129  slurmstepd: error: *** STEP 3082812.1 ON g13 CANCELLED AT 2025-10-20T20:38:36 ***
 130  srun: error: g13: task 0: Terminated
 131  srun: Force Terminated StepId=3082812.1
```
