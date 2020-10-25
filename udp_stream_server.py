import argparse
import cv2
import keras

from ipc.ipc_manager import IpcManager
from processing_utils.haar_detector import FaceDetector
from processing_utils.cropper import ImgCropper
from classifier.inference import predict_signle_image


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0,
                        help="Id source webcam")
    parser.add_argument("--show", type=int, default=1,
                        help="Show video stream")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd_args()
    cap = cv2.VideoCapture(args.source)
    face_det = FaceDetector()
    border = (20, 20)
    img_sz = (64, 64)
    cropper = ImgCropper(border=(20, 20))
    classifier = keras.models.load_model("classifier/checkpoints/model_ep10.h5")
    print("Webcam: ", args.source)
    print("Show stream: ", args.show)
    ipc = IpcManager(shmem_name="/udp_streamer_shmem", sem_name="/udp_streamer", mq_name="/udp_streamer")
    i = 0
    while True:
        print("Frame # ", i)
        ret, frame = cap.read()
        processed = frame.copy()
        if args.show:
            cv2.imshow('Source', frame)
        try:
            boxes = face_det.detect(frame)
            for (x, y, w, h) in boxes:
                cv2.rectangle(processed, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_img = cv2.resize(cropper((x, y, w, h), frame), img_sz)
                prediction = predict_signle_image(classifier, face_img, {0: "Female", 1: "Male"})
                best = max(prediction, key=prediction.get)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(best) + ": " + str(int(prediction[best] * 100.0)) + "%"
                cv2.putText(processed, text, (x, y-10), font, 1, (0, 255, 0))
            if args.show:
                cv2.imshow('Processed', processed)
            ipc.write_frame(processed)
        except Exception as e:
            print("Processing exception.", e, " Writing raw frame")
            ipc.write_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    if args.show:
        cv2.destroyAllWindows()