
# In[3]:


import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# In[4]:

import torch

torch.cuda.empty_cache()

# In[5]:




# In[6]:


import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'MAX_SPLIT_SIZE_1=32MB'

# In[7]:



# In[8]:


# Here is how to change the GAN model
# gan_model_path = '/content/CLIP_Steering/stylegan2-ada-pytorch/pretrained/ffhq.pkl'

gan_model_path = '/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/stylegan/ClothingGAN/CLIP_Steering/pretrained/ffhq.pkl'
# Here is how we specify the desired attributes
attributes = ["a cartoon cat ", "a cat"]
# attributes = ["an old face", "a young face", "a happy face","a sad face"]
class_index = 0  # which attribute do we want to maximize, ie. attribute number 0, or attribute number 1 etc.

# Here is where to store the checkpoints i.e. weights
checkpoint_dir = f'checkpoints/results_maximize_{attributes[class_index]}_probability'

# In[9]:


import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as torch_transforms

import os
import pathlib


sys.path.append(os.path.abspath(os.getcwd()))
# sys.path.append('/content/CLIP_Steering/stylegan2-ada-pytorch/')

sys.path.append(
    '/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/stylegan/CLIP_Steering/stylegan2-ada-pytorch/')

import torch_utils
import ganalyze_transformations as transformations
import ganalyze_common_utils as common
from clip_classifier_utils import SimpleTokenizer

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_save_image(img, out_dir):
    img = (gan_images.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    img_np = img.detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    for i in range(6):  # By defualt batchsize= 6
        row = i // 3
        col = i % 3
        ax[row, col].imshow(img_np[i])
    plt.show()

    for i in range(6):
        filename = f"image_{batch_start}_{i}.png"
        plt.imsave(f"{checkpoint_dir}/{filename}", img_np[i])


def gan_output_transform(imgs):
    # Input:
    # img: NCHW
    #
    # Output
    # img_np: HWC RGB image

    imgs = (imgs * 127.5 + 128).clamp(0, 255).float()
    return imgs


def clip_input_transform(images):
    # Input
    # img_np: torch tensor of shape NHWC, RGB image
    #
    # Output
    # image_input: torch tensor of shape NHWC

    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)

    transform = torch.nn.Sequential(
        torch_transforms.Resize((256, 256)),
        torch_transforms.CenterCrop((224, 224)),
        torch_transforms.Normalize(image_mean, image_std),
    )

    return transform(images)


def get_clip_scores(image_inputs, encoded_text, model, class_index=0):
    # TODO: clarify class index
    image_inputs = clip_input_transform(image_inputs).to(device)
    image_feats = model.encode_image(image_inputs).float()
    image_feats = F.normalize(image_feats, p=2, dim=-1)

    similarity_scores = torch.matmul(image_feats, torch.transpose(encoded_text, 0, 1))
    similarity_scores = similarity_scores.to(device)
    return similarity_scores.narrow(dim=-1, start=class_index, length=1).squeeze(dim=-1)


def get_clip_probs(image_inputs, encoded_text, model, class_index=0):
    image_inputs = clip_input_transform(image_inputs).to(device)
    image_feats = model.encode_image(image_inputs).float()
    image_feats = F.normalize(image_feats, p=2, dim=-1)

    clip_probs = (100.0 * torch.matmul(image_feats, torch.transpose(encoded_text, 0, 1))).softmax(dim=-1)
    clip_probs = clip_probs.to(device)

    return clip_probs.narrow(dim=-1, start=class_index, length=1).squeeze(dim=-1)





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
    batch_size=5_000,  # style layer quite small

)

inst = get_instrumented_model(config.model, config.output_class,
                              config.layer, torch.device('cuda'), use_w=config.use_w, training=True)
model = inst.model
# model.
from matplotlib import pyplot as plt



w_l = model.sample_latent(1, seed=5).detach().cpu().numpy()
w1 = [w_l] * model.get_max_latents()  # one per layer

image = model.sample_torch(w_l)

# In[12]:


latent_space_dim = model.get_latent_dims()
print(latent_space_dim)



# Set up clip classifier
clip_model_path = '/home/aj/.cache/clip/ViT-B-32.pt'
clip_model = torch.jit.load(clip_model_path)
clip_model.eval()
clip_model.to(device)
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

tokenizer = SimpleTokenizer("CLIP/clip/bpe_simple_vocab_16e6.txt.gz")
sot_token = tokenizer.encoder['<|startoftext|>']
eot_token = tokenizer.encoder['<|endoftext|>']
text_descriptions = [f"This is a photo of {label}" for label in attributes]
text_tokens = [[sot_token] + tokenizer.encode(desc) + [eot_token] for desc in text_descriptions]
text_inputs = torch.zeros(len(text_tokens), clip_model.context_length, dtype=torch.long)

for i, tokens in enumerate(text_tokens):
    text_inputs[i, :len(tokens)] = torch.tensor(tokens)

# These are held constant through the optimization, akin to labels
text_inputs = text_inputs.to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_inputs).float()
    text_features = F.normalize(text_features, p=2, dim=-1)
text_features.to(device)

# In[15]:


# Setting up Transformer, i.e. the function that transforms the input z vector
# --------------------------------------------------------------------------------------------------------------
transformer_params = ['OneDirection', 'None']
transformer = transformer_params[0]
transformer_arguments = transformer_params[1]
if transformer_arguments != "None":
    key_value_pairs = transformer_arguments.split(",")
    key_value_pairs = [pair.split("=") for pair in key_value_pairs]
    transformer_arguments = {pair[0]: pair[1] for pair in key_value_pairs}
else:
    transformer_arguments = {}

transformation = getattr(transformations, transformer)(latent_space_dim, vocab_size, **transformer_arguments)
transformation = transformation.to(device)

# function that is used to score the (attribute, image) pair
scoring_fun = get_clip_probs

# In[19]:


# Training
# --------------------------------------------------------------------------------------------------------------
# optimizer
# loss_n = torch.nn.MSELoss().requires_grad_(True)
loss_n = torch.nn.MSELoss().requires_grad_(True)

optimizer = torch.optim.Adam(transformation.parameters(), lr=0.0002)  # as specified in GANalyze
losses = common.AverageMeter(name='Loss')

#  training settings
optim_iter = 0
batch_size = 6  # Do not change
train_alpha_a = -0.4  # Lower limit for step sizes
train_alpha_b = 0.4  # Upper limit for step sizes
num_samples = 450  # Number of samples to train for # Ganalyze uses 400,000 samples. Use smaller number for testing.

# create training set
# np.random.seed(seed=0)
truncation = 0.4
# zs = common.truncated_z_sample(num_samples, latent_space_dim, truncation)
zs = model.sample_latent(num_samples, seed=5).detach().cpu().numpy()

# checkpoint_dir = f'checkpoints/results_maximize_{attributes[class_index]}_probability'
pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# loop over data batches
for batch_start in range(0, num_samples, batch_size):

    # input batch
    s = slice(batch_start, min(num_samples, batch_start + batch_size))
    z = torch.from_numpy(zs[s]).type(torch.FloatTensor).to(device)
    print(z)
    y = None

    # step_sizes = (train_alpha_b - train_alpha_a)*np.ones(batch_size)*0.0001
    # print(step_sizes)

    step_sizes = (train_alpha_b - train_alpha_a) * np.random.random(
        size=(batch_size)) + train_alpha_a  # sample step_sizes

    step_sizes_broadcast = np.repeat(step_sizes, latent_space_dim).reshape([batch_size, latent_space_dim])
    step_sizes_broadcast = torch.from_numpy(step_sizes_broadcast).type(torch.FloatTensor).to(device)

    # ganalyze steps
    #     gan_images = G(z, None)
    gan_images = model.sample_torch(z)
    gan_images = gan_output_transform(gan_images)
    out_scores = scoring_fun(
        image_inputs=gan_images, encoded_text=text_features, model=clip_model, class_index=class_index,
    )
    # TODO: ignore z vectors with less confident clip scores
    target_scores = out_scores + torch.from_numpy(step_sizes).to(device).float().requires_grad_(True)

    z_transformed = transformation.transform(z, None, step_sizes_broadcast)
    gan_images_transformed = model.sample_torch(z_transformed)
    #     gan_images_transformed = G(z_transformed, None)
    gan_images_transformed = gan_output_transform(gan_images_transformed).to(device)
    out_scores_transformed = scoring_fun(
        image_inputs=gan_images_transformed, encoded_text=text_features, model=clip_model, class_index=class_index,
    ).to(device).float().requires_grad_(True)
    print(out_scores_transformed.requires_grad_(True))
    # print(target_scores)

    # compute loss
    loss = loss_n(out_scores_transformed, target_scores)
    #     loss = loss_function(prediction, torch.tensor([[10.0,31.0]]).double()).float()
    #     input('lllllll')

    # backwards
    # loss.backward()
    optimizer.step()

    # print loss
    losses.update(loss.item(), batch_size)
    if optim_iter % 10 == 0:
        logger.info(
            f'[Maximizing score for {attributes[class_index]}] Progress: [{batch_start}/{num_samples}] {losses}')
        print(f'[Maximizing score for {attributes[class_index]}] Progress: [{batch_start}/{num_samples}] {losses}')

    if optim_iter % 50 == 0:
        logger.info(f"saving checkpoint at iteration {optim_iter}")
        print(f"saving checkpoint at iteration {optim_iter}")
        torch.save(transformation.state_dict(),
                   os.path.join(checkpoint_dir, "pytorch_model_{}.pth".format(batch_start)))

        # plot and save sample images
        plot_save_image(gan_images, checkpoint_dir)

    optim_iter = optim_iter + 1

# In[20]:


# In[17]:
#
#
# # Testing
import torch_utils
import ganalyze_transformations as transformations
import ganalyze_common_utils as common
from clip_classifier_utils import SimpleTokenizer


# --------------------------------------------------------------------------------------------------------------
# TODO Figure out where to resume checkpoints

def one_hot(index, vocab_size=1000):
    output = torch.zeros(index.size(0), vocab_size).to(index.device)
    output.scatter_(1, index.unsqueeze(-1), 1)
    return output


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1) * 255


# helper function for visualization of test images
# def make_image(z, y, step_size, transform):
def make_image(z, step_size, transform):
    if transform:
        z_transformed = transformation.transform(z, None, step_size)
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        z = z_transformed

    # gan_images = utils.pytorch.denorm(generator(z, y))
    # gan_images = common.pytorch.denorm(G(z, y))
    # gan_images = G(z, None)
    gan_images = denorm(model.sample_torch(z))
    gan_images_np = gan_images.permute(0, 2, 3, 1).detach().cpu().numpy()
    gan_images = gan_output_transform(gan_images)

    # gan_images = gan_images.view(-1, *gan_images.shape[-3:])
    gan_images = gan_images.to(device)

    # out_scores_current = output_transform(assessor(gan_images))
    # out_scores_current = gan_output_transform(scoring_fun(gan_images))
    # TODO Check assessor section in Ganalyze test script line 106
    out_scores_current = scoring_fun(image_inputs=gan_images, encoded_text=text_features, model=clip_model,
                                     class_index=class_index).to(device).float()
    out_scores_current = out_scores_current.detach().cpu().numpy()
    if len(out_scores_current.shape) == 1:
        out_scores_current = np.expand_dims(out_scores_current, 1)

    return (gan_images_np, z, out_scores_current)
#
#
# # In[18]:
#
#
# # Test settings
#
# # code from ganalyze_commons_utils added below
#
import numpy as np
from scipy.stats import truncnorm
import PIL.ImageDraw
import PIL.ImageFont


def truncated_z_sample(batch_size, dim_z, truncation=1):
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z))
    return truncation * values


def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        imarray = np.uint8(imarray)
        # raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows * H, cols * W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid


def annotate_outscore(array, outscore):
    for i in range(array.shape[0]):
        I = PIL.Image.fromarray(np.uint8(array[i, :, :, :]))
        draw = PIL.ImageDraw.Draw(I)
        # font = PIL.ImageFont.truetype("arial.ttf", int(array.shape[1]/8.5))
        font = PIL.ImageFont.load_default()
        message = str(round(np.squeeze(outscore)[i], 2))
        x, y = (0, 0)
        w, h = font.getsize(message)
        # print(w, h)
        draw.rectangle((x, y, x + w, y + h), fill='white')
        draw.text((x, y), message, fill="black", font=font)
        array[i, :, :, :] = np.array(I)
    return (array)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# # Test settings
#
num_samples = 10
truncation = 1
iters = 3
# np.random.seed(seed=999) #removed like training code block
annotate = True
vocab_size = 1  # vocab size is one for debugging. else change to:
# vocab_size = clip_model.vocab_size.item()

if vocab_size == 0:
    num_categories = 1
else:
    num_categories = vocab_size

for y in range(num_categories):

    ims = []
    outscores = []

    # zs = utils.common.truncated_z_sample(num_samples, dim_z, truncation)
    # zs = common.truncated_z_sample(num_samples, latent_space_dim, truncation)
    # zs = truncated_z_sample(num_samples, latent_space_dim, truncation)
    zs=model.sample_latent(num_samples, seed=5).detach().cpu().numpy()

    ys = np.repeat(y, num_samples)
    zs = torch.from_numpy(zs).type(torch.FloatTensor).to(device)
    ys = torch.from_numpy(ys).to(device)
    ys = one_hot(ys, vocab_size)
    # step_sizes = np.repeat(np.array(opts["alpha"]), num_samples * dim_z).reshape([num_samples, dim_z])

    alpha = 0.2
    step_sizes = np.repeat((alpha), num_samples * latent_space_dim).reshape([num_samples, latent_space_dim])

    # TODO instead write a loop here to sample values of alpha
    # alpha= [-0.5,-0.25,0,0.25,0.5]
    # step_sizes = np.repeat(np.array[of alpha], num_samples * latent_space_dim).reshape([num_samples, latent_space_dim])

    step_sizes = torch.from_numpy(step_sizes).type(torch.FloatTensor).to(device)
    feed_dicts = []
    for batch_start in range(0, num_samples, 4):
        s = slice(batch_start, min(num_samples, batch_start + 4))
        # feed_dicts.append({"z": zs[s], "y": ys[s], "truncation": truncation, "step_sizes": step_sizes[s]})
        feed_dicts.append({"z": zs[s], "truncation": truncation, "step_sizes": step_sizes[s]})

    for feed_dict in feed_dicts:
        ims_batch = []
        outscores_batch = []
        z_start = feed_dict["z"]

        step_sizes = feed_dict["step_sizes"]

        # if opts["mode"] == "iterative":
        # choose from iterative or bigger_step
        mode = "iterative"
        if mode == "iterative":
            print("iterative")

            # original seed image
            # x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
            x, tmp, outscore = make_image(feed_dict["z"], feed_dict["step_sizes"], transform=False)

            x = np.uint8(x)
            if annotate:
                # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                # ims_batch.append(common.annotate_outscore(x, outscore))
                ims_batch.append(annotate_outscore(x, outscore))


            else:
                if annotate:
                    # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                    # ims_batch.append(common.annotate_outscore(x, outscore))
                    ims_batch.append(annotate_outscore(x, outscore))


                else:
                    ims_batch.append(x)
            outscores_batch.append(outscore)

            # negative clone images
            z_next = z_start
            step_sizes = -step_sizes
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes
                feed_dict["z"] = z_next
                # x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                x, tmp, outscore = make_image(feed_dict["z"], feed_dict["step_sizes"], transform=True)
                x = np.uint8(x)
                z_next = tmp
                if annotate:
                    # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                    # ims_batch.append(common.annotate_outscore(x, outscore))
                    ims_batch.append(annotate_outscore(x, outscore))

                else:
                    if annotate:
                        # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                        ims_batch.append(common.annotate_outscore(x, outscore))

                    else:
                        ims_batch.append(x)
                outscores_batch.append(outscore)

            ims_batch.reverse()

            # positive clone images
            step_sizes = -step_sizes
            z_next = z_start
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes
                feed_dict["z"] = z_next

                # x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                x, tmp, outscore = make_image(feed_dict["z"], feed_dict["step_sizes"], transform=True)

                x = np.uint8(x)
                z_next = tmp

                if annotate:
                    # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                    ims_batch.append(annotate_outscore(x, outscore))

                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

        else:
            print("bigger_step")

            # original seed image
            # x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=False)
            x, tmp, outscore = make_image(feed_dict["z"], feed_dict["step_sizes"], transform=False)
            x = np.uint8(x)
            if annotate:
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
            else:
                ims_batch.append(x)
            outscores_batch.append(outscore)

            # negative clone images
            step_sizes = -step_sizes
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes * (iter + 1)

                # x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                x, tmp, outscore = make_image(feed_dict["z"], feed_dict["step_sizes"], transform=True)

                x = np.uint8(x)

                if annotate:
                    # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                    ims_batch.append(annotate_outscore(x, outscore))
                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

            ims_batch.reverse()
            outscores_batch.reverse()

            # positive clone images
            step_sizes = -step_sizes
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes * (iter + 1)

                # x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                x, tmp, outscore = make_image(feed_dict["z"], feed_dict["step_sizes"], transform=True)
                x = np.uint8(x)
                if annotate:
                    # ims_batch.append(utils.common.annotate_outscore(x, outscore))
                    ims_batch.append(annotate_outscore(x, outscore))

                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

        ims_batch = [np.expand_dims(im, 0) for im in ims_batch]
        ims_batch = np.concatenate(ims_batch, axis=0)
        ims_batch = np.transpose(ims_batch, (1, 0, 2, 3, 4))
        ims.append(ims_batch)

        outscores_batch = [np.expand_dims(outscore, 0) for outscore in outscores_batch]
        outscores_batch = np.concatenate(outscores_batch, axis=0)
        outscores_batch = np.transpose(outscores_batch, (1, 0, 2))
        outscores.append(outscores_batch)

    ims = np.concatenate(ims, axis=0)
    outscores = np.concatenate(outscores, axis=0)
    ims_final = np.reshape(ims, (ims.shape[0] * ims.shape[1], ims.shape[2], ims.shape[3], ims.shape[4]))
    # I = PIL.Image.fromarray(utils.common.imgrid(ims_final, cols=iters * 2 + 1))
    I = PIL.Image.fromarray(common.imgrid(ims_final, cols=iters * 2 + 1))

    # TODO change the code below to write a new file for every result cycle. Currently it over writes the my_results.jpg file.
    # For now, make sure to download your result "my_results.jpg" before running again
    result_dir = "my_results"
    I.save(os.path.join(result_dir + ".jpg"))

    # I.save(os.path.join(result_dir, categories[y] + ".jpg"))
    print("y: ", y)
#
# # In[ ]:
#
#
#
