import streamlit as st
import numpy as np
from model import Encoder, Decoder
from skimage import io, img_as_float
from model import Encoder, Decoder
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
st.title('Deep Learning based Image Steganography')

def normalize_img(img):
  img = (img-np.min(img.flatten()))/(np.max(img.flatten())-np.min(img.flatten()))
  return img

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

col1, col2 = st.columns(2)

with col1:

  cover_img  = st.file_uploader("Upload cover image", type=['png','jpg','jpeg','TIFF'])
  if cover_img is not None:
    c_img = io.imread(cover_img)
    cover_img =  img_as_float(c_img)
    cover_img = normalize_img(cover_img)#Adding to avoid division by zero

    st.image(cover_img)

with col2:
  secret_img  = st.file_uploader("Upload secret image", type=['png','jpg','jpeg','TIFF'])
  if secret_img is not None:
    s_img = io.imread(secret_img)
    secret_img =  img_as_float(s_img)
    secret_img = normalize_img(secret_img) #Adding to avoid division by zero

    st.image(secret_img)

encoder = Encoder()
decoder = Decoder()

encoder_path = "./1encoder.pt"
decoder_path = "./1decoder.pt"

encoder.load_state_dict(torch.load(encoder_path, map_location = device))
decoder.load_state_dict(torch.load(decoder_path, map_location = device))

# print(secret_img.shape)
if secret_img is not None and cover_img is not None:
  p = transform(s_img)
  s = transform(c_img)
  s = torch.unsqueeze(s,0)
  p = torch.unsqueeze(p,0)
  
  enc_output = encoder(s,p)
  
  stegano_img = enc_output[0,:,:,:].permute(1,2,0).cpu()
  stegano_img = stegano_img.detach().numpy()
  stegano_img = normalize_img(stegano_img) #Adding to avoid division by zero
  st.image(stegano_img)
  # plt.imshow(stegano_img.detach().numpy())

  dec_output = decoder(enc_output)

  output_img = dec_output[0,:,:,:].permute(1,2,0).cpu()
  output_img = output_img.detach().numpy()
  output_img = normalize_img(output_img) #Adding to avoid division by zero
  st.image(output_img)






