from ipc.posix_shmem import PosixShmem
from ipc.posix_mqueue import PosixMQueue
from ipc.posix_semaphore import PosixSemaphore

import cv2
import numpy as np

SHMEM_DEFAULT_SIZE = 2500 * 2000 * 3


class IpcManager:

    def __init__(self, shmem_name, sem_name, mq_name):
        self._shmem = PosixShmem(shmem_name, SHMEM_DEFAULT_SIZE)
        self._sem = PosixSemaphore(sem_name)
        self._mq = PosixMQueue(mq_name)

    def __str__(self):
        return "StreamerIpcManager"

    def __call__(self, *args, **kwargs):
        pass

    def write_frame(self, frame):
        assert type(frame) == np.ndarray
        ysz, xsz, csz = frame.shape
        msg = str(ysz * xsz * csz) + " " + str(16) + " " + str(ysz) + " " + str(xsz)
        if self._sem.lock(timeout=0.010):
            self._shmem.write(frame)
            self._sem.unlock()
            self._mq.send(msg)

    def read_frame(self):
        msg = self._mq.receive(timeout=1.0)
        if msg is None:
            return None
        msg = msg[:-1].decode("utf-8")
        self._sem.lock(timeout=0.010)
        args = msg.split()
        total, img_type, ysz, xsz = int(args[0]), int(args[1]), int(args[2]), int(args[3])
        data = self._shmem.read()
        img = data[:total].copy()
        img = img.reshape((ysz, xsz, 3))
        self._sem.unlock()
        return img