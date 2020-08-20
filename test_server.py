import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.joinpath('ds/proto')))
import ds.evolver as e
