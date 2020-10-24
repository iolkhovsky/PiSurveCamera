import socket
import time
import imagezmq
import cv2
import argparse
from processing_utils.haar_detector import FaceDetector


def parse():
    parser = argparse.ArgumentParser(description='Camera streamer')
    parser.add_argument('--client', metavar='N', type=str, default='igor-msi',
                    help='Name of client PC')
    parser.add_argument('--raw_stream', metavar='R', type=int, default=0,
                    help='Sent raw stream')
    parser.add_argument('--cam_id', metavar='C', type=int, default=0,
                    help='Sent raw stream')
    return parser.parse_args()

    
def main():
    args = parse()
    client_address = f'tcp://{args.client}:5555'
    sender = imagezmq.ImageSender(connect_to=client_address)
    print("Going to stream to", client_address)
    rpi_name = socket.gethostname()
    print("from", rpi_name)
    face_det = FaceDetector()
    cap = cv2.VideoCapture(args.cam_id)
    print("Camera id:", args.cam_id)
    last_timestamp = time.time()
    frame_cnt = 0
    detections_cnt = 0
    while True:  
        ret, frame = cap.read()
        frame_cnt += 1
        boxes = face_det.detect(frame)
        if len(boxes):
            detections_cnt += 1
        for idx, (x, y, w, h) in enumerate(boxes):
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if args.raw_stream or len(boxes):
            sender.send_image(rpi_name, frame)

        if (frame_cnt + 1) % 10 == 0:
            interval = time.time() - last_timestamp
            print("Average fps:", int(frame_cnt / (interval + 1e-9)), "frames", frame_cnt, "detections", detections_cnt)
            detections_cnt = 0


if __name__ == "__main__":
    main()
