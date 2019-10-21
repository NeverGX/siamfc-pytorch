import numpy as np
import time
import cv2
import glob
import os
from config import config
from tracker import SiamFCTracker
import argparse
import time
from region_to_bbox import region_to_bbox

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/wangkh/tracker_benchmark/data')
parser.add_argument('--model_dir', type=str, default='/home/wangkh/ex/models/siamfc.pth')
arg = parser.parse_args()
data_dir = arg.data_dir
model_dir = arg.model_dir


def run_SiamFC():


    videolist = os.listdir(data_dir)
    videolist.sort()
    nv = len(videolist)
    precision_all = 0
    iou_all = 0
    fps_all = 0

    for video in videolist:
        gt, frame_name_list, n_frames = _init_video(data_dir, video)
        bboxs = np.zeros((n_frames, 4))
        tracker = SiamFCTracker(model_dir, config.gpu_id)
        start = time.time()
        for i in range(n_frames):
            frame = cv2.imread(frame_name_list[i], cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if i==0:
                tracker.init(frame, gt[0])
                bbox = gt[0]
                bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            else:
                bbox = tracker.update(frame)

            bboxs[i] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            # visualization
            # cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0),thickness=2)
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            # cv2.imshow('img', frame)
            # cv2.waitKey(30)

            end = time.time()
            fps = 1/ ((end-start)/n_frames)
        _, precision, _, iou = compile_results(gt, bboxs, 20)
        print("video " + ' -- ' + str(video) + "  Precision: " + str(precision) + "  IOU: " + str(iou) + "  FPS: " + str(fps))
        precision_all = precision_all + precision
        iou_all = iou_all + iou
        fps_all = fps_all + fps

    P = precision_all / nv
    I = iou_all / nv
    F = fps_all / nv
    print("averge_Precision:" + str(P))
    print("averge_IOU:" + str(I))
    print("averge_FPS:" + str(F))


def _init_video(datapath, video):

    frame_name_list = glob.glob(os.path.join(datapath, video)+'/img/**.jpg')
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

def compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01
    return iou

def compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')
    return dist



if __name__ == '__main__':
    run_SiamFC()
