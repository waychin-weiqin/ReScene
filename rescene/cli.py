"""Lightweight CLI; CUDA dependencies load only in the requested pipeline."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['synthetic', 'real', 'check'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help(); return
    command = parser.parse_args(argv[:1]).command
    scripts = {'synthetic':'reproduce_multistate_hard.py', 'real':'reproduce_real_multistate.py', 'check':'check_reproduction.py'}
    script = ROOT/'rescene/pipeline'/scripts[command]
    subprocess.run([sys.executable,str(script),*argv[1:]],cwd=ROOT,check=True)
