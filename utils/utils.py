import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2
from pathlib import Path

module_logger = logging.getLogger(__name__)


def get_data_path():
    return Path(__file__).resolve().parent.parent / "data"

def encode_vid_H264(vid_path):
    """
    Re-encode the video using the H.264 encoding
    """
    import subprocess
    base_name = vid_path.split(".")
    cmd = "/bin/ffmpeg -hide_banner -loglevel error -i %s -c:v libx264 %s" % \
        (vid_path, base_name[0]+".mp4")
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def video_from_imgs(imgs, video_path, fps=60, encode_H264=True):
    """
    Build a video using the imgs in the given array.
    imgs: NxHxW or NxWxHxC images (double or uint8)
    video_path: path to the output video (do not include extension)
    """
    if type(imgs) == torch.Tensor:
        imgs = imgs.detach().numpy()
    elif type(imgs) != np.ndarray:
        module_logger.error("imgs must be a tensor or ndarray")
        return False

    if imgs.ndim != 3 and imgs.ndim != 4:
        module_logger.error("imgs must either be 3- or 4-dimensional")
        return False

    if imgs.dtype != np.float64 and imgs.dtype != np.float32 and \
        imgs.dtype != np.uint8:
        module_logger.error("imgs datatype must be float, double, or uint8")
        return False

    if imgs.ndim == 4 and imgs.shape[3] != 3:
        module_logger.error("Color shold be the last channel, axis=3")
        return False

    # Convert to uint8 if applicable
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8)

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    avi_video_path = str(video_path) + ".avi"
    video = cv2.VideoWriter(avi_video_path, fourcc, fps,
                            (imgs.shape[2], imgs.shape[1]), True)
    N = imgs.shape[0]
    for i in range(N):
        if i % 1000 == 0:
            print("%d / %d" % (i, N))
        img = imgs[i]
        if img.ndim == 2:
            img = np.tile(img[:,:,None], (1, 1, 3))
        video.write(img)

    video.release()

    if encode_H264:
        encode_vid_H264(avi_video_path)
        os.remove(avi_video_path)

    return True

def trim_video(video_path, trim_video_path, start_frame, end_frame,
               encode_H264=False):
    """
    Trim the video between the start and end frames. Use to reduce video file
    sizes for PowerPoint.
    """
    cap = cv2.VideoCapture(video_path)

    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(trim_video_path, fourcc, fps, (width, height),
                             True)

    for i in range(N):
        ret_val, frame = cap.read()
        if not ret_val:
            module_logger.error("Error reading from video")
            return
        if i >= start_frame and i < end_frame:
            writer.write(frame)

    writer.release()

    if encode_H264:
        encode_vid_H264(trim_video_path)
        os.remove(trim_video_path)

def mmm(x):
    if isinstance(x, torch.Tensor):
        return (x.min().item(), x.to(torch.float64).mean().item(), x.max().item())
    elif isinstance(x, np.ndarray):
        return (x.min(), x.mean(), x.max())
    else:
        return None

def whos(x):
    assert type(x) == np.ndarray or type(x) == torch.Tensor
    if type(x) == np.ndarray:
        return (x.shape, x.dtype)
    else:
        return (x.shape, x.dtype, x.device)

def keep_top_k_percent(x, k):
    """
    Keeps the top k percent of the elements in x and zeros the remaining
    elements.
    """
    assert k >= 0 and k <= 1

    x_sorted = np.sort(x.ravel())
    threshold = x_sorted[np.round((1 - k) * len(x.ravel())).astype(np.int64)]
    x_top_k = x.copy()
    x_top_k[x_top_k < threshold] = 0

    return x_top_k

def upsample_nn(img, target_r_size=800):
    D = target_r_size // img.shape[0]
    if D < 1:
        return img
    return cv2.resize(img, (img.shape[1] * D, img.shape[0] * D), 0, 0,
                      interpolation=cv2.INTER_NEAREST)


def imshow(img, title=None, color_flag=None, **imshow_args):
    """
    Display the image.
    """
    if type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()
        
    plt.figure()
    if color_flag is not None:
        img = cv2.cvtColor(img, color_flag)
    plt.imshow(img, **imshow_args)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def list_of_dicts_to_dict(lod):
    D = {}
    for d in lod:
        D.update(d)
    return D

        
def gamma_correct(img, gamma=2.2):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.dtype == np.float32 or img.dtype == np.float64:
        return img**(1/gamma)

    # Use a LUT for fixed precision
    if img.dtype == np.uint8:
        M = 255
    elif img.dtype == np.uint16:
        M = 65535
    
    x = np.arange(M + 1)
    y = ((x.astype(np.float64) / M)**(1/gamma)*M).astype(img.dtype)

    img_gc = np.take(y, img)
    return img_gc