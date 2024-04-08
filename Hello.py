import streamlit as st
import pandas as pd
from io import StringIO
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Kiểm tra xem có GPU có sẵn không và đặt PyTorch sử dụng GPU nếu có
device = 'cpu'

# Load mô hình ResNet pre-trained
model = models.resnet18()

# Thay thế lớp cuối cùng
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Chuyển mô hình đến GPU nếu có sẵn
model = model.to(device)

model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))

# Định nghĩa các biến đổi
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về kích thước 224x224
    transforms.ToTensor(),  # Chuyển ảnh về dạng tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
])

classes = ['146.000', '195.000', '244.000', '293.000', '391.000', '489.000', '588.000', '686.000', '784.000', '980.000']

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

    # Print weights of the first layer
    first_layer_weights = next(model.parameters()).data
    # st.write(first_layer_weights.shape)
    # st.write(first_layer_weights)
    
    # Áp dụng softmax để chuyển đổi đầu ra thành xác suất
    probabilities = F.softmax(output, dim=1)
    
    # st.write(output)

    # img = input_tensor.cpu().numpy().transpose((1, 2, 0))
    # img = std * img + mean  # unnormalize
    # img = np.clip(img, 0, 1)
    # plt.imshow(img)
    # st.pyplot(plt)
    with col2:
        # for i in range(10):
        #     text = f'{classes[i]}: {probabilities[0][i].item()*100:.2f}%'
        #     st.progress(probabilities[0][i].item(), text=text)
            # st.write(classes[i], ':', "{:.2f}%".format(probabilities[0][i].item()*100))
        time.sleep(1)
        st.write('# Predicted:', classes[torch.argmax(probabilities, dim=1)], 'VND')