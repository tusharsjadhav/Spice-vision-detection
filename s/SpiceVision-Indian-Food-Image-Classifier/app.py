import torch
import torch.nn as nn
import timm
from torchvision import transforms
import streamlit as st
from PIL import Image
import numpy as np

classes = ['Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Sandwich', 
           'Taco', 'Taquito', 'apple_pie', 'burger', 'butter_naan', 'chai', 
           'chapati', 'cheesecake', 'chicken_curry', 'chole_bhature', 
           'dal_makhani', 'dhokla', 'fried_rice', 'ice_cream', 'idli', 
           'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 
           'momos', 'omelette', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 
           'samosa', 'sushi']

class food_classifier(nn.Module):
    def __init__(self, num_classes=34):
        super(food_classifier, self).__init__()
        self.base_model = timm.create_model("resnet50", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        resnet_out_size = 2048
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(resnet_out_size, num_classes))

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

model = food_classifier()

model_state_dict = torch.load('models/food_image_classification_model.pth')
model.load_state_dict(model_state_dict)

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.set_page_config(page_title="SpiceVision")

image = st.sidebar.file_uploader('UPLOAD THE IMAGE')

if st.sidebar.button("identify"):
    st.image(image)

    with st.spinner("🔍"):
        img, image_tensor = preprocess_image(image, transform)
        probs = predict(model, image_tensor)
        idx = probs.argmax()
        out = classes[idx]
        st.write(f"The item in the image is a {out}")

st.sidebar.markdown("---")
st.sidebar.caption("Created By: Sidhant Manale")
