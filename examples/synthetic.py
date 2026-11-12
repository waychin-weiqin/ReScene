"""Prepare, smoke-test, train and report synthetic Multi-State."""
import sys
from rescene.cli import main
if __name__ == '__main__': main(['synthetic', *sys.argv[1:]])
