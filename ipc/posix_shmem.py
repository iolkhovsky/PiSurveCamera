import posix_ipc
import mmap
import numpy as np


class PosixShmem:

    def __init__(self, name, sz):
        assert type(name) == str
        assert type(sz) == int
        self._name = name
        self._size = sz
        self._shmem = posix_ipc.SharedMemory(name, flags=posix_ipc.O_RDWR + posix_ipc.O_CREAT, mode=0o666,
                                            read_only=False, size=self._size)
        self._fd = self._shmem.fd
        self._mapfile_in = mmap.mmap(self._fd, self._size, prot=mmap.PROT_READ)
        self._mapfile_out = mmap.mmap(self._fd, self._size, prot=mmap.PROT_WRITE)

    def write(self, data2write):
        assert type(data2write) == np.ndarray
        data2write = data2write.tobytes()
        self._mapfile_out.seek(0)
        if data2write is not None:
            self._mapfile_out.write(data2write)

    def read(self):
        self._mapfile_in.seek(0)
        data = self._mapfile_in.read(self._size)
        data = np.frombuffer(data, dtype=np.uint8)
        return data


if __name__ == "__main__":
    shm = PosixShmem("/udp_streamer_shmem", 2500*2000*3)
    data = shm.read()
    print(data.shape)
    if len(data):
        print(data[:min(10, len(data))])
    shm.write(np.fromstring("Shmem agile test"))
    data = shm.read()
    test_str = data[:20].tobytes().decode("utf-8")
    print(test_str)