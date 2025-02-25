module purge
module load conda
module load gcc
conda create --name AIM python=3.11.4
conda activate AIM
python -m ensurepip --upgrade
pip install -r requirements.txt
