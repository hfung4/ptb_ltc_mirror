# Databricks notebook source
!pip install pytest pytest-cov

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pytest
import os
import sys
from pathlib import Path

# COMMAND ----------

# Get path of the `run_unit_tests` notebook
notebook_path = Path(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
# Go to repo root
package_root = notebook_path.parent.parent
# Set current directory to repo root
os.chdir(f'/Workspace/{package_root}')
%pwd

# COMMAND ----------

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True

# COMMAND ----------

# Run pytest from notebook/script using the pytest.main() method
retcode = pytest.main([".", "-p", "no:cacheprovider"])

# COMMAND ----------

# Fail the cell execution if we have any test failures.
assert retcode == 0, 'The pytest invocation failed. See the log above for details.'
