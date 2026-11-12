"""Cross-process GPU leases for the original and sequential LTGS queues."""
import fcntl
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 3


def acquire_gpu(gpu, free, directory=None):
    """Return an open exclusive lease, or None; keep it for the entire task."""
    if not isinstance(gpu,int) or gpu<0:
        raise ValueError('GPU index must be a nonnegative integer')
    directory = Path(directory or ROOT / 'artifacts/ltgs-gpu-leases')
    directory.mkdir(parents=True, exist_ok=True)
    lease = (directory / f'gpu{gpu}.lock').open('a+')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    try:
        if free(gpu):
            return lease
    except BaseException:
        lease.close()
        raise
    lease.close()
    return None


def require_compatible_peer(lock_path, state_path):
    """Fail closed around old live runners, including across PID namespaces."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = json.loads(Path(state_path).read_text())
            if state.get('protocol', {}).get('scheduler_version') != VERSION:
                raise RuntimeError('Legacy peer runner is active. Coordinated host-side '
                                   'handoff is required; do not run mixed schedulers.')


def ordered_jobs(scenes):
    return [(scene, setting) for setting in ('sparse3', 'dense') for scene in scenes]


def initial_ready(scene, original):
    work = Path(original) / 'dense' / scene
    return ((work / 'initial-train.done.json').is_file() and
            (work / 'initial_model/point_cloud/iteration_30000/point_cloud.ply').is_file())
