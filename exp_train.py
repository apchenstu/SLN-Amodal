import os

cmd = """
python amodal_train.py evaluate \
 --dataset ./datasets/D2S \
 --model ./checkpoints/D2SA.pth \
 --limit -1\
"""

os.system(cmd)
