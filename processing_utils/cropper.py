import cv2

from common_utils.utils import *


class ImgCropper:

    def __init__(self, border=(0, 0)):
        self._border = border

    def _extend_box(self, box):
        x, y, w, h = box
        border_x, border_y = self._border
        center_x, center_y = 0.5 * (2 * x + w - 1), 0.5 * (2 * y + h - 1)
        w_ext = w * (1.0 + border_x / 100.)
        h_ext = h * (1.0 + border_y / 100.)
        x_ext = center_x - 0.5 * w_ext
        y_ext = center_y - 0.5 * h_ext
        return int(x_ext), int(y_ext), int(w_ext), int(h_ext)

    def _crop(self, box, img):
        box = self._extend_box(box)
        x, y, w, h = validate_box(box, xsz=img.shape[1], ysz=img.shape[0])
        assert 2 <= len(img.shape) <= 3
        return img[y:y+h, x:x+w].copy()

    def process(self, boxes, img):
        if type(boxes) == tuple:
            return self._crop(boxes, img)
        boxes = list(boxes)
        return [self._crop(x, img) for x in boxes]

    def __call__(self, *args, **kwargs):
        return self.process(args[0], args[1])
