import argparse
import cv2
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from implementations.context_encoder.datasets import *
from implementations.context_encoder.models import *
import face_recognition

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--save_model_interval", type=int, default=10000,
                    help="interval between model saves")
parser.add_argument('--dont_load_latest_gan', dest='load_latest_gan',
                    action='store_false',
                    help="train gan from scrath; if not flagged, load the "
                         "latest gan in saved_models folder")
parser.set_defaults(load_latest_gan=True)

opt = parser.parse_args()
print(opt)

def load_latest_gan():
    all_saved_models = os.listdir("saved_models")
    all_saved_models.sort(key=lambda x: int(x))
    newest_gan_index = all_saved_models[-1]

    print(f"loading GAN from batch number: {newest_gan_index}")

    generator_path = os.path.join("saved_models",
                                  f"{newest_gan_index}",
                                  f"g_{newest_gan_index}")
    discriminator_path = os.path.join("saved_models",
                                  f"{newest_gan_index}",
                                  f"d_{newest_gan_index}")

    loaded_generator = Generator(channels=opt.channels)
    loaded_generator.load_state_dict(torch.load(generator_path))

    loaded_discriminator = Discriminator(channels=opt.channels)
    loaded_discriminator.load_state_dict(torch.load(discriminator_path))
    return loaded_generator, loaded_discriminator
g, d = load_latest_gan()
###############################################################################
############################# GOOD STUFF START HERE ###########################
###############################################################################


from PIL import Image

original_image = Image.open('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
manpulated_image = Image.open('/mnt/data/deepfakes/manipulated_sequences'
                            '/face_crops/018_019/frame108.png')
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

original_resized = transform(original_image)
def get_face_locations(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (128, 128))
    return face_recognition.face_locations(resized)

face_locations = get_face_locations('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
# face_locations = get_face_locations('/mnt/data/deepfakes/manipulated_sequences'
#                             '/face_crops/018_019/frame108.png')
bounding_box = face_locations[0]

def fix_bounding_box(bounding_box, target_h=64, target_w=64):
    top, right, bottom, left = bounding_box
    current_h = bottom - top
    current_w = right - left
    if current_h > target_h:
        diff = current_h - target_h
        bottom -= diff
    else:
        diff = target_h - current_h
        top -= diff
    if current_w > target_w:
        diff = current_w - target_w
        right -= int(np.floor(diff / 2))
        left += int(np.ceil(diff / 2))
    else:
        diff = target_w - current_w
        right += int(np.floor(diff / 2))
        left -= int(np.ceil(diff / 2))

    return (top, right, bottom, left)

bounding_box = fix_bounding_box(bounding_box)
top, right, bottom, left = bounding_box

original_resized[..., top:bottom, left:right] = 1

inpt = torch.tensor(original_resized)
inpt = inpt.unsqueeze(0)



cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
masked_samples = Variable(inpt.type(Tensor))
g = g.cuda()
gen_mask = g(masked_samples.cuda())
filled_samples = masked_samples.clone()
filled_samples[:, :, top:bottom, left:right] = gen_mask


original_image = Image.open('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
manpulated_image = Image.open('/mnt/data/deepfakes/manipulated_sequences/face_crops/018_019/frame108.png')

transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

original_resized = transform(original_image)
original_resized = original_resized.unsqueeze(0)
original_tensor = Variable(original_resized.type(Tensor))

sample = torch.cat((masked_samples.data, filled_samples.data,
                    original_tensor.data), -2)
save_image(sample, "sandbox_images/original_reconstructed_images"
                   "/018_reconstructed.png",
           nrow=1,
           normalize=True)


###############################################################################
############################ CODE DUPE HERE !!!! ##############################
###############################################################################


from PIL import Image

original_image = Image.open('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
manpulated_image = Image.open('/mnt/data/deepfakes/manipulated_sequences'
                            '/face_crops/018_019/frame108.png')
original_image = manpulated_image
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

original_resized = transform(original_image)
def get_face_locations(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (128, 128))
    return face_recognition.face_locations(resized)

face_locations = get_face_locations('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
# face_locations = get_face_locations('/mnt/data/deepfakes/manipulated_sequences'
#                             '/face_crops/018_019/frame108.png')
bounding_box = face_locations[0]

def fix_bounding_box(bounding_box, target_h=64, target_w=64):
    top, right, bottom, left = bounding_box
    current_h = bottom - top
    current_w = right - left
    if current_h > target_h:
        diff = current_h - target_h
        bottom -= diff
    else:
        diff = target_h - current_h
        top -= diff
    if current_w > target_w:
        diff = current_w - target_w
        right -= int(np.floor(diff / 2))
        left += int(np.ceil(diff / 2))
    else:
        diff = target_w - current_w
        right += int(np.floor(diff / 2))
        left -= int(np.ceil(diff / 2))

    return (top, right, bottom, left)

bounding_box = fix_bounding_box(bounding_box)
top, right, bottom, left = bounding_box

original_resized[..., top:bottom, left:right] = 1

inpt = torch.tensor(original_resized)
inpt = inpt.unsqueeze(0)



cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
masked_samples = Variable(inpt.type(Tensor))
g = g.cuda()
gen_mask_MANIP = g(masked_samples.cuda())
filled_samples = masked_samples.clone()
filled_samples[:, :, top:bottom, left:right] = gen_mask_MANIP


original_image = Image.open('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
manpulated_image = Image.open('/mnt/data/deepfakes/manipulated_sequences/face_crops/018_019/frame108.png')
original_image = manpulated_image
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

original_resized = transform(original_image)
original_resized = original_resized.unsqueeze(0)
original_tensor = Variable(original_resized.type(Tensor))

sample = torch.cat((masked_samples.data, filled_samples.data,
                    original_tensor.data), -2)
save_image(sample, "sandbox_images/original_reconstructed_images"
                   "/018_019_reconstructed2.png",
           nrow=1,
           normalize=True)

import scipy.misc

diff = abs(gen_mask_MANIP - gen_mask)
diff_map = 255.0 * diff / diff.max()
diff = diff.squeeze()
diff = diff.permute(1, 2, 0)
diff_map_np = diff.cpu().detach().numpy()
scipy.misc.imsave('/home/uriel/dev/PyTorch-GAN/implementations/context_encoder/sandbox_images/original_reconstructed_images/diff_pristine_fake_face.png', diff_map_np)


###############################################################################
############################ SECOND CODE DUPE HERE !!!! #######################
###############################################################################


from PIL import Image

original_image = Image.open('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
manpulated_image = Image.open('/mnt/data/deepfakes/manipulated_sequences'
                            '/face_crops/018_019/frame108.png')
original_image = manpulated_image
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

original_resized = transform(original_image)
def get_face_locations(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (128, 128))
    return face_recognition.face_locations(resized)

face_locations = get_face_locations('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
face_locations = get_face_locations('/mnt/data/deepfakes/manipulated_sequences'
                            '/face_crops/018_019/frame108.png')
bounding_box = face_locations[0]

def fix_bounding_box(bounding_box, target_h=64, target_w=64):
    top, right, bottom, left = bounding_box
    current_h = bottom - top
    current_w = right - left
    if current_h > target_h:
        diff = current_h - target_h
        bottom -= diff
    else:
        diff = target_h - current_h
        top -= diff
    if current_w > target_w:
        diff = current_w - target_w
        right -= int(np.floor(diff / 2))
        left += int(np.ceil(diff / 2))
    else:
        diff = target_w - current_w
        right += int(np.floor(diff / 2))
        left -= int(np.ceil(diff / 2))

    return (top, right, bottom, left)

bounding_box = fix_bounding_box(bounding_box)
top, right, bottom, left = bounding_box

original_resized[..., top:bottom, left:right] = 1

inpt = torch.tensor(original_resized)
inpt = inpt.unsqueeze(0)



cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
masked_samples = Variable(inpt.type(Tensor))
g = g.cuda()
gen_mask_MANIP = g(masked_samples.cuda())
filled_samples = masked_samples.clone()
filled_samples[:, :, top:bottom, left:right] = gen_mask_MANIP


original_image = Image.open('/mnt/data/deepfakes/original_sequences/face_crops/018/frame108.png')
manpulated_image = Image.open('/mnt/data/deepfakes/manipulated_sequences/face_crops/018_019/frame108.png')
original_image = manpulated_image
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

original_resized = transform(original_image)
original_resized = original_resized.unsqueeze(0)
original_tensor = Variable(original_resized.type(Tensor))

sample = torch.cat((masked_samples.data, filled_samples.data,
                    original_tensor.data), -2)
save_image(sample, "sandbox_images/original_reconstructed_images"
                   "/018_019_reconstructed.png",
           nrow=1,
           normalize=True)

import scipy.misc

diff = abs(gen_mask_MANIP - gen_mask)
diff_map = 255.0 * diff / diff.max()
diff = diff.squeeze()
diff = diff.permute(1, 2, 0)
diff_map_np = diff.cpu().detach().numpy()
scipy.misc.imsave('/home/uriel/dev/PyTorch-GAN/implementations'
                  '/context_encoder/sandbox_images/original_reconstructed_images/diff_pristine_fake_face2.png', diff_map_np)
