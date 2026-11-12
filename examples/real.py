"""Prepare, smoke-test, train and report the four accepted real scenes."""
import sys
from rescene.cli import main
if __name__ == '__main__': main(['real', *sys.argv[1:]])
