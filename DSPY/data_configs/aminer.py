import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile = f"{dataroot}/aminer.pt"

dataset = "aminer"
testlength = 3
vallength = 3
