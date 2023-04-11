#@title Load Model
selected_model = 'lookbook'

# Load model
from IPython.utils import io
import torch
import PIL
import numpy as np
import ipywidgets as widgets
from PIL import Image
import imageio
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config
from skimage import img_as_ubyte

# Speed up computation
torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# Specify model to use
config = Config(
  model='StyleGAN2',
  layer='style',
  output_class=selected_model,
  components=80,
  use_w=True,
  batch_size=5_000, # style layer quite small

)

inst = get_instrumented_model(config.model, config.output_class,
                              config.layer, torch.device('cpu'), use_w=config.use_w,training=True)
model = inst.model
# model.
from matplotlib import pyplot as plt
# im1 = model.()

w_l = model.sample_latent(1, seed=5).detach().cpu().numpy()
print(w_l)
w1 = [w_l]*model.get_max_latents() # one per layer

import cv2
im1 = model.sample_np(w1)
im2 = model.sample_np(w_l)
image=model.sample_torch(w_l)
print(image)
com=np.hstack((im1,im2))
# cv2.imwrite('new_i.png',com[:,:,::-1]*255)
plt.imshow(com)
plt.show()

# loss = torch.nn.MSELoss().requires_grad_(True)
# print(loss)
# input_ = torch.randn(3, 5).to('cuda').float()
# print(input_.requires_grad_(True))
# target = torch.randn(3, 5).to('cuda').float()
# print(target)
# output = loss(input_, target)
# output.backward()