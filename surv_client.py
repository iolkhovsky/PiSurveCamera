import cv2
import imagezmq


def main():
    image_hub = imagezmq.ImageHub()
    while True:
        rpi_name, image = image_hub.recv_image()
        print("Received frame from", rpi_name)
        cv2.imshow(rpi_name, image)
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')


if __name__ == "__main__":
    main()