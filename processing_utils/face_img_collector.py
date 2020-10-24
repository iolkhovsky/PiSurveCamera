import argparse
import cv2
import logging
import time
from os.path import join, isdir
from os import makedirs
from shutil import rmtree
import sys

sys.path.append(join(sys.path[0], "../"))

from processing_utils.haar_detector import FaceDetector
from processing_utils.cropper import ImgCropper
from common_utils.utils import *


def parse():
    parser = argparse.ArgumentParser(description="Stream face image collector")
    parser.add_argument("--camid", type=int, default=0, help="Id of webcamera")
    parser.add_argument("--border_x", type=float, default=20, help="Extension border size X (% from width)")
    parser.add_argument("--border_y", type=float, default=20, help="Extension border size Y (% from height)")
    parser.add_argument("--repo", type=str, default="collected_data_repo", help="Path to repository")
    parser.add_argument("--time_limit", type=str, default="60m", help="Time limit for application")
    parser.add_argument("--count_limit", type=int, default=10000, help="Count limit for images")
    parser.add_argument("--min_sample_time", type=str, default="1s", help="Min time between samples")
    return parser.parse_args()


def main(args):
    logging.basicConfig(filename="face_img_collector.log", format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(args2str(args))
    logging.info("Application start")
    print(args2str(args))
    if isdir(args.repo):
        logging.info("Removing existing file repository")
        rmtree(args.repo)
    makedirs(args.repo)
    logging.info("Creating new file repository")
    face_det = FaceDetector()
    logging.info("Creating new face detector")
    cropper = ImgCropper(border=(args.border_x, args.border_y))
    cap = cv2.VideoCapture(args.camid)
    logging.info("Creating new CV capturer")
    samples_cnt = 0
    startup_stamp = time.time()
    time_limit_sec = decode_time_str(args.time_limit)
    save_min_period = decode_time_str(args.min_sample_time)
    last_save_tstamp = time.time()
    logging.info("Start collecting...")
    while (time.time() < startup_stamp + time_limit_sec) and (samples_cnt < args.count_limit):
        ret, frame = cap.read()
        if time.time() > last_save_tstamp + save_min_period:
            boxes = face_det.detect(frame)
            if len(boxes):
                subframes = cropper(boxes, frame)
                for subid, subfr in enumerate(subframes):
                    filename = join(args.repo, str(samples_cnt + subid) + ".jpg")
                    cv2.imwrite(filename, subfr)
                samples_cnt += len(boxes)
                last_save_tstamp = time.time()
                logging.info("Collected: "+str(samples_cnt))
                print("Collected: ", samples_cnt)


if __name__ == "__main__":
    main(parse())
