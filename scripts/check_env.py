import pandas as pd
import numpy as np
try:
    import fastf1
    print(f"FastF1 version: {fastf1.__version__}")
except ImportError:
    print("FastF1 not installed yet")

print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

try:
    import pyspark
    print(f"PySpark version: {pyspark.__version__}")
except ImportError:
    print("PySpark not installed yet")
