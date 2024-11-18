import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device) 
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    # 读取图像并转换为灰度
    img = nib.load(image_path).dataobj
    img = img[:, :, 5]  # 假设只处理一张slice
    img = Image.fromarray(img)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # 增加batch维度
    return img_tensor

def predict(model, img_tensor, device):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return pred

def save_prediction(pred, output_path):
    plt.imsave(output_path, pred, cmap='gray')

def main(image_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    img_tensor = preprocess_image(image_path)
    pred = predict(model, img_tensor, device)
    save_prediction(pred, output_path)
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    image_path = "dataset/testing/patient101/patient101_frame01.nii.gz"
    model_path = "models/model-unetpp-0.012335.pt"
    output_path = "pred.png"
    main(image_path, model_path, output_path) 