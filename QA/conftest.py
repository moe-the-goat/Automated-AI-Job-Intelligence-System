"""pytest path setup. Lets `import core_ai` etc. work from within QA/ tests
when running via `python -m pytest QA/`. The standalone `run_all.py` does
its own path setup, so this is only relevant when pytest is installed.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
