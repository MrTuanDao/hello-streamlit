import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write('Labels') 

    # To read file as bytes:
    st.image(uploaded_file)

    st.write('Data')

import torch
from torchvision import models

# Kiểm tra xem có GPU có sẵn không và đặt PyTorch sử dụng GPU nếu có
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load mô hình ResNet pre-trained
resnet = models.resnet18()

# Chuyển mô hình đến GPU nếu có sẵn
resnet = resnet.to(device)

