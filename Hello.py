import streamlit as st
import pandas as pd
from io import StringIO
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Kiểm tra xem có GPU có sẵn không và đặt PyTorch sử dụng GPU nếu có
device = 'cpu'

# Load mô hình ResNet pre-trained
resnet = models.resnet18()

# Thay thế lớp cuối cùng
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)

# Chuyển mô hình đến GPU nếu có sẵn
resnet = resnet.to(device)

resnet.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))

model = resnet

# Định nghĩa các biến đổi
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về kích thước 224x224
    transforms.ToTensor(),  # Chuyển ảnh về dạng tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
])

classes = ['VND146.000', 'VND195.000', 'VND244.000', 'VND293.000', 'VND391.000', 'VND489.000', 'VND588.000', 'VND686.000', 'VND784.000', 'VND980.000']

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

    with col1:
        # To read file as bytes:
        st.image(uploaded_file)

    # Convert the file-like object to an image
    image = Image.open(uploaded_file)

    # Tiền xử lý hình ảnh
    input_tensor = transform(image)

    # Tạo một batch nhỏ bằng cách thêm một chiều
    input_batch = input_tensor.unsqueeze(0)

    # Chuyển batch đến thiết bị phù hợp
    input_batch = input_batch.to(device)

    # Đưa hình ảnh qua mô hình để dự đoán
    with torch.no_grad():
        output = model(input_batch)

    # Áp dụng softmax để chuyển đổi đầu ra thành xác suất
    probabilities = F.softmax(output, dim=1)
    
    with col2:
        for i in range(10):
            st.write(classes[i], ':', "{:.2f}%".format(probabilities[0][i].item()*100))