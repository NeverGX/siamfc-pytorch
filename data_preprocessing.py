import cv2
import numpy as np
import argparse
import glob
import os
from tqdm import tqdm
from concurrent import futures
from config import config

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default='/home/username/Downloads/got-10k/full_data/val', help='got_10k data dir')
parser.add_argument('--output_dir',type=str, default='/home/username/Downloads/got-10k/crop_data', help='got_10k output dir')
parser.add_argument('--num_processings',type=int, default=None)
arg = parser.parse_args()
datapath = arg.data_dir
outputpath = arg.output_dir
num_processings = arg.num_processings
cfg = config

def _init_video(video):

    frame_name_list = glob.glob(os.path.join(datapath, video)+'/**.jpg')
    frame_name_list.sort()

    try:
        gt_file = os.path.join(datapath, video, 'groundtruth.txt')
        try:
            gt = np.loadtxt(gt_file, dtype=float, delimiter=',')
        except:
            gt = np.loadtxt(gt_file, dtype=float)
    except:
        gt_file = os.path.join(datapath, video, 'groundtruth_rect.txt')
        try:
            gt = np.loadtxt(gt_file, dtype=float, delimiter=',')
        except:
            gt = np.loadtxt(gt_file, dtype=float)

    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'
    return gt, frame_name_list, n_frames

def get_center(x):
    return (x - 1.) / 2.

def xyxy2cxcywh(bbox):
    return get_center(bbox[0]+bbox[2]), \
           get_center(bbox[1]+bbox[3]), \
           (bbox[2]-bbox[0]), \
           (bbox[3]-bbox[1])

def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    xmin = cx - original_sz // 2
    xmax = cx + original_sz // 2
    ymin = cy - original_sz // 2
    ymax = cy + original_sz // 2
    im_h, im_w, _ = img.shape

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]
    if left != 0 or right !=0 or top!=0 or bottom!=0:
        if img_mean is None:
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=img_mean)
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch

def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x
    instance_img = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    return instance_img, scale_x, s_x

def crop_video(video):

    gt, frame_name_list, n_frames = _init_video(video)
    savepath = os.path.join(outputpath, video)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for i in range(n_frames):
        img = cv2.imread(frame_name_list[i])
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        bbox = gt[i]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]
        instance_img, _, _ = get_instance_image(img, bbox,
                                                cfg.exemplar_size, cfg.instance_size, cfg.context_amount,
                                                img_mean)
        cv2.imwrite(savepath+"/{:0>8d}.x.jpg".format(i+1), instance_img)
def main():
    videos = os.listdir(datapath)
    videos.sort()
    # crop_video(videos[0])
    with futures.ProcessPoolExecutor(max_workers=None) as executor:
        fs = [executor.submit(crop_video, video) for video in videos]
        bar = tqdm(futures.as_completed(fs), total=len(videos))
        for f in bar:
            bar.set_description('Loading:')
        #or in this way
        # results = executor.map(crop_video,videos)
if __name__ == '__main__':
    main()
