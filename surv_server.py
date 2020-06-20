import socket
import time
import imagezmq
import cv2
import argparse
from haar_detector import FaceDetector

def parse():
    parser = argparse.ArgumentParser(description='Camera streamer')
    parser.add_argument('--client', metavar='N', type=str, default='igor-msi',
                    help='Name of client PC')
    parser.add_argument('--raw_stream', metavar='C', type=int, default=0,
                    help='Sent raw stream')
    return parser.parse_args()

    
def main():
    args = parse()
    sender = imagezmq.ImageSender(connect_to= f'tcp://{args.client}:5555')
    rpi_name = socket.gethostname()
    face_det = FaceDetector()
    cap = cv2.VideoCapture(0)
    while True:  
        ret, frame = cap.read()
        boxes = face_det.detect(frame)
        for idx, (x, y, w, h) in enumerate(boxes):
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if args.raw_stream or len(boxes):
            sender.send_image(rpi_name, frame)


if __name__ == "__main__":
    main()
