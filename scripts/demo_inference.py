"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time
import math

import cv2
import numpy as np
import torch
from tqdm import tqdm
import natsort
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.writer import DataWriter

current_directory = os.path.dirname(__file__)
print(current_directory)


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=False,
                    default=
                        current_directory+ "/../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=False,
                    default=current_directory+ "/../pretrained_models/fast_res50_256x192.pth",
                    help='checkpoint file name')
parser.add_argument('--sp', default=True, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory',default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory')
parser.add_argument('--save_img', default=True, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=True)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

# folder dataset
# args.inputimg = "C:/Users/Lenovo/Desktop/KI6/PBL5/TestImages3/192x256/CR_1_0093.jpg"
# # folder lưu kết quả
# args.outputpath = "C:/Users/Lenovo/Desktop/KI6/PBL5/TestImages3/192x256"

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(
    ',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device(
    "cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError(
                'Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError(
                'Error: --detfile must refer to a detection json file, not directory.')

    # for images

    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):

        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]
        print(im_names)
        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


def set_args(input_path,output_path):
    args.inputimg = input_path
    args.outputpath = output_path
def load_detector(arguments):
    mode, input_source = check_input()

    det_loader = DetectionLoader(input_source,get_detector(arguments), cfg, arguments, batchSize=arguments.detbatch, mode=mode,
                                 queueSize=arguments.qsize)
    det_worker = det_loader.start()
    return det_loader,det_worker
def load_model(arguments,config):
    pose_model = builder.build_sppe(config.MODEL, preset_cfg=config.DATA_PRESET)

    print('Loading pose model from %s...' % (arguments.checkpoint,))
    model = torch.load(
        arguments.checkpoint, map_location=arguments.device)

    pose_model.load_state_dict(model)
    pose_dataset = builder.retrieve_dataset(config.DATASET.TRAIN)

    if len(arguments.gpus) > 1:
        pose_model = torch.nn.DataParallel(
            pose_model, device_ids=arguments.gpus).to(arguments.device)
    else:
        pose_model.to(arguments.device)
    return pose_model,pose_dataset

def detection(detector,pose_model,pose_dataset ):
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    # current_millis = round(time.time() * 1000)

    pose_model.eval()
    # print("Load model2 duration: " + str(round(time.time() * 1000) - current_millis))
    # current_millis = round(time.time() * 1000)
    # Init data writer
    queueSize = args.qsize
    writer = DataWriter(cfg, args, save_video=False,
                        queueSize=queueSize).start()

    data_len = detector.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    print("ne",im_names_desc,"ne")
    batchSize = args.posebatch


    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores,
                 ids, cropped_boxes) = detector.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None,
                                None, orig_img, im_name)
                    continue
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j *
                                  batchSize:min((j + 1) * batchSize, datalen)]

                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm,
                            cropped_boxes, orig_img, im_name)


        print_finish_info()
        while (writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' +
                  str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        detector.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            detector.terminate()
            while (writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' +
                      str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues
            detector.terminate()
            writer.terminate()
            writer.clear_queues()
            detector.clear_queues()
    return  writer.final_result_1
    # print("Load detection duration: " + str(round(time.time() * 1000) - current_millis))

# set_args("C:/Users/Lenovo/Desktop/KI6/PBL5/TestImages3/192x256/CR_1_0096.jpg","C:/Users/Lenovo/Desktop/KI6/PBL5/TestImages3/192x256")
#

current_millis = round(time.time() * 1000)
pose_model_1, pose_dataset_1 = load_model(args, cfg)
print("Load model duration: " + str(round(time.time() * 1000)-current_millis))
#
# current_millis = round(time.time() * 1000)
# decorator_1 ,dec_worker_1= load_detector(args)
# print("Load detetor duration: " + str(round(time.time() * 1000)-current_millis))
#
# current_millis = round(time.time() * 1000)
# detection(decorator_1,pose_model_1,pose_dataset_1)
# print("Detection: " + str(round(time.time() * 1000)-current_millis))















# if __name__ == "__main__":
#     print("START")
#
#     # mode, input_source = check_input()
#
#     if not os.path.exists(args.outputpath):
#         os.makedirs(args.outputpath)
#     current_millis = round(time.time() * 1000)
#     det_loader = load_detector(args)
#     det_worker = det_loader.start()
#
#     print("Load detector duration: " + str(round(time.time() * 1000)-current_millis))
#
#     # Load pose model
#     current_millis = round(time.time() * 1000)
#     pose_model ,pose_dataset= load_model(args,cfg)
#     print("Load model duration: " + str(round(time.time() * 1000) - current_millis))
#
#     current_millis = round(time.time() * 1000)
#
#     pose_model.eval()
#     print("DONE 3")
#     print("Load model duration: " + str(round(time.time() * 1000) - current_millis))
#     current_millis = round(time.time() * 1000)
#     # Init data writer
#     queueSize = args.qsize
#     writer = DataWriter(cfg, args, save_video=False,
#                             queueSize=queueSize).start()
#
#     data_len = det_loader.length
#     im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
#
#     batchSize = args.posebatch
#     # print("DONE 2")
#     # print("Load model duration: " + str(round(time.time() * 1000)-current_millis))
#     current_millis =  round(time.time() * 1000)
#
#     runtime_profile = {
#         'dt': [],
#         'pt': [],
#         'pn': []
#     }
#     if args.flip:
#         batchSize = int(batchSize / 2)
#     try:
#         for i in im_names_desc:
#             start_time = getTime()
#             with torch.no_grad():
#                 (inps, orig_img, im_name, boxes, scores,
#                  ids, cropped_boxes) = det_loader.read()
#                 if orig_img is None:
#                     break
#                 if boxes is None or boxes.nelement() == 0:
#                     writer.save(None, None, None, None,
#                                 None, orig_img, im_name)
#                     continue
#                 # if args.profile:
#                 #     ckpt_time, det_time = getTime(start_time)
#                 #     runtime_profile['dt'].append(det_time)
#                 # Pose Estimation
#                 inps = inps.to(args.device)
#                 datalen = inps.size(0)
#                 leftover = 0
#                 if (datalen) % batchSize:
#                     leftover = 1
#                 num_batches = datalen // batchSize + leftover
#                 hm = []
#                 for j in range(num_batches):
#                     inps_j = inps[j *
#                                   batchSize:min((j + 1) * batchSize, datalen)]
#                     if args.flip:
#                         inps_j = torch.cat((inps_j, flip(inps_j)))
#                     hm_j = pose_model(inps_j)
#                     if args.flip:
#                         hm_j_flip = flip_heatmap(
#                             hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
#                         hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
#                     hm.append(hm_j)
#                 hm = torch.cat(hm)
#                 hm = hm.cpu()
#                 writer.save(boxes, scores, ids, hm,
#                             cropped_boxes, orig_img, im_name)
#                 if args.profile:
#                     ckpt_time, post_time = getTime(ckpt_time)
#                     runtime_profile['pn'].append(post_time)
#
#             if args.profile:
#                 # TQDM
#                 im_names_desc.set_description(
#                     'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
#                         dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
#                 )
#         print_finish_info()
#         while (writer.running()):
#             time.sleep(1)
#             print('===========================> Rendering remaining ' +
#                   str(writer.count()) + ' images in the queue...', end='\r')
#         writer.stop()
#         det_loader.stop()
#     except Exception as e:
#         print(repr(e))
#         print('An error as above occurs when processing the images, please check it')
#         pass
#     except KeyboardInterrupt:
#         print_finish_info()
#         # Thread won't be killed when press Ctrl+C
#         if args.sp:
#             det_loader.terminate()
#             while (writer.running()):
#                 time.sleep(1)
#                 print('===========================> Rendering remaining ' +
#                       str(writer.count()) + ' images in the queue...', end='\r')
#             writer.stop()
#         else:
#             # subprocesses are killed, manually clear queues
#
#             det_loader.terminate()
#             writer.terminate()
#             writer.clear_queues()
#             det_loader.clear_queues()
#
#     print("Detection duration: " + str(round(time.time() * 1000)-current_millis))
#
