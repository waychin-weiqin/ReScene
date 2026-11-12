"""Validate and aggregate evaluated ReScene histories; no baseline evaluation."""
import argparse
from rescene.cli import main
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',choices=['synthetic','real'],required=True)
    p.add_argument('--scenes',nargs='+',help='The same ordered scene selection used during preparation')
    a=p.parse_args()
    main([a.subset,'report',*(['--scenes',*a.scenes] if a.scenes else [])])
