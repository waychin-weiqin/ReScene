"""Explicit experiment scope and bounded retention for LTGS WAT jobs."""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def policy(root=ROOT):
    p = Path(root) / 'artifacts/ltgs/run-policy.json'
    if p.exists():
        result = json.loads(p.read_text())
        for key in ('original_settings', 'sequential_settings'):
            assert result[key] and set(result[key]) <= {'dense', 'sparse3'}, key
        return result
    return {'original_settings': ['sparse3', 'dense'], 'sequential_settings': ['sparse3', 'dense'],
            'min_free_gib': 100, 'cleanup_validated_caches': False}


def require_space(root=ROOT):
    required = policy(root)['min_free_gib'] * 2**30
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeError(f'LTGS storage guard: {free / 2**30:.1f} GiB free; '
                           f'{required / 2**30:.0f} GiB reserve required. Preserve checkpoints; reclaim disposable caches.')


def wait_stage(proc, log, root=ROOT):
    """Monitor disk reserve while a stage runs in its own process group."""
    while True:
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                require_space(root)
            except RuntimeError as exc:
                log.write('\n' + str(exc) + '\n'); log.flush()
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try: os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    proc.wait()
                # Even a child handling SIGTERM with exit 0 must not receive a done marker.
                return 75


def cleanup_cache(work, scene, summary, root=ROOT):
    """Delete only a completed task's MASt3R tree, never models or snapshots."""
    if not policy(root).get('cleanup_validated_caches'): return
    work = Path(work)
    d = json.loads(Path(summary).read_text())
    assert d['scene'] == scene and d['heldout_rgb_exact_match'] is True and d['test_views'] > 0
    assert all(math.isfinite(d['period_macro'][m]) for m in ('PSNR', 'SSIM', 'LPIPS'))
    assert (work / 'update.done.json').is_file(), 'No successful model update marker'
    cache = work / 'output' / scene / 'mast3r'
    if not cache.exists(): return
    assert not cache.is_symlink() and cache.resolve().is_relative_to(work.resolve()), 'Unsafe cache path'
    size = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file() and not p.is_symlink())
    record = {'path': str(cache), 'bytes_before': size, 'validated_summary': str(summary),
              'time': datetime.now(timezone.utc).isoformat(), 'status': 'deleting',
              'retained': 'models, temporal checkpoints, immutable snapshots, masks, logs, metrics, stage markers'}
    record_path = work / 'cache-retention.json'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    shutil.rmtree(cache)
    record['status'] = 'deleted'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
