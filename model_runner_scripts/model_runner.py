import subprocess
import itertools
from time import sleep


def test_ses():
    number_of_runs = 1
    
    for _ in range(number_of_runs):
        python_call = ["python3", "-m", "models.model_tester"]
        process = subprocess.Popen(python_call)
        process.wait()
        sleep(5)

if __name__ == "__main__":
    test_ses()
