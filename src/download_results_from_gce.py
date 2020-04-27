"""Download paper results from GCE."""
import subprocess
from pathlib import Path

project_id = "neat-glazing-257206"
local_path = str(Path('.').resolve().parent)
gs_path = "trial:/home/elpistolero317/workspace/asymmetric-tri-rec/paper_results"

if __name__ == "__main__":
    subprocess.run(['gcloud', 'compute', 'scp', '--project',
                    project_id, '--recurse', gs_path, local_path])
