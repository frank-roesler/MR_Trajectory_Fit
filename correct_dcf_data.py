# import torch
# from params import *
# import matplotlib.pyplot as plt
# from time import time
# from glob import glob
# import numpy as np


# train_data = glob("train_data/*")
# for step in range(len(train_data)):
#     filename = train_data[step]
#     rosette_batch, dcf_batch = torch.load(train_data[step])
#     rosette_batch = rosette_batch[:, :-1, :]
#     dcf_batch = dcf_batch[:, :-1]
#     # torch.save((rosette_batch, dcf_batch), filename)
