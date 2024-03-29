import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch


def get_cam(model, target_layers, use_cuda=True):
    return GradCAM(model=model, target_layers=target_layers, use_cuda=True)

def show_cam_on_image(img, mask, alpha=1.0):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def plot_cam(cam_obj, misclassified_images_dict, classes, keys_list=None):
    fig, axs = plt.subplots(20, 2, figsize=(10,60), squeeze=False)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    idx = 0

    for i in range(20):
        idx = next(keys_list)
        true_label = classes[misclassified_images_dict[idx][1]]
        pred_label = classes[misclassified_images_dict[idx][2].item()]

        img2 = misclassified_images_dict[idx][0].unsqueeze(dim=0)

        true_label_idx = misclassified_images_dict[idx][2]
        target = [ClassifierOutputTarget(true_label_idx)]

        grayscale_cam = cam_obj(input_tensor=img2, targets=target)
        grayscale_cam = grayscale_cam[0, :]
        cam = show_cam_on_image(np.transpose(img2.squeeze(), (1,2,0)), grayscale_cam)
        axs[i, 0].imshow(np.transpose(img2.squeeze(), (1,2,0)), cmap='gray', interpolation='bilinear')
        axs[i, 1].imshow(cam, interpolation='bilinear')

        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'True Label: {true_label}\n Pred Label:{pred_label}')
        
        
        
def plot_imgs(classes, misclassified_imgs_dict, imgs=20):
    fig, axs = plt.subplots(5, 4, figsize=(10, 10), squeeze=False)
    fig.tight_layout(h_pad=2)
    idx = 0
    key_list = iter(list(misclassified_imgs_dict.keys()))

    for i in range(5):
        for j in range(4):
            idx = next(key_list)
            img = misclassified_imgs_dict[idx][0]
            true_label = classes[misclassified_imgs_dict[idx][1]]
            pred_label = classes[misclassified_imgs_dict[idx][2].item()]

            axs[i, j].imshow(np.transpose(img, (1,2,0)), cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f'True Label: {true_label}\n Pred Label:{pred_label}')
            
def get_misclassified_images(model, test_loader, device):
  model.eval()
  idx = 0
  misclassified_imgs = {}
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output  = model(data)
      pred = output.argmax(dim=1, keepdim=True)

      for sample in range(data.shape[0]):
        if (target[sample] != pred[sample]):
            misclassified_imgs[idx] = [data[sample].cpu(), target[sample].cpu(), pred[sample].cpu()]
        idx += 1
  return misclassified_imgs