from collections.abc import Iterable
import spektral as sp
import numpy as np
from scipy import sparse
from itertools import product


def bounds(obj: Iterable) -> np.ndarray:
    max_arr = np.max(obj, axis=1)
    min_arr = np.min(obj, axis=1)
    return np.array([min_arr, max_arr])


def quadrant_k(obj, k) -> np.ndarray:
    return obj[:k, k:]


def encode_pmsp(jobs, n_job_classes, machines, class_delays, t):
    jobs_reps = np.array([[j.work_time, j.work_load, j.arrival_time,
                               *[1 if i == j.job_class else 0 for i in range(n_job_classes)]] for j in jobs])
    machines_reps = np.array(
        [[m.get_remaining_job_time(t),
              *[1 if i == m.last_run_job_class else 0 for i in range(n_job_classes)]] for m
        in machines])

    jobs_padded = np.pad(jobs_reps, ((0, 0), (0, machines_reps.shape[1])))
    machines_padded = np.pad(machines_reps, ((0, 0), (jobs_reps.shape[1], 0)))

    n_jobs = len(jobs)
    n_machines = len(machines)
    n_nodes = n_jobs + n_machines

    n_edges = n_jobs * n_machines

    x = np.concatenate([jobs_padded, machines_padded])
    x = np.pad(x, ((0, 0), (0, 1)))
    x[:n_jobs, -1] = 1

    a = np.zeros((n_nodes, n_nodes))
    a[:n_jobs, n_jobs:] = 1
    a = sparse.coo_array(a)

    edges = []
    for p in product(jobs, machines):
        delay = 0
        job = p[0]
        machine = p[1]
        if machine.last_run_job_class is not None:
            delay = class_delays[job.job_class, machine.last_run_job_class]
        edges.append(delay)
    e = np.array(edges).reshape(len(edges), 1)

    encoding = sp.data.Graph(
        x=x.astype(np.float32),
        a=a.astype(np.float32),
        e=e.astype(np.float32),
        y=np.array([0, 1, 2]).astype(np.float32),
        u=0
    )
    return encoding
