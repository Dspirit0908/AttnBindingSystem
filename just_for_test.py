# coding: utf-8

import os
import json
import time
import torch
import random
import datetime
import numpy as np

x = torch.randn([2, 3])
print(x)
print(torch.softmax(x, dim=1))
