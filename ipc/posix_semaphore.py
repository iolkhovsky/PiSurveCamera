import posix_ipc


class PosixSemaphore:

    def __init__(self, name):
        flags = posix_ipc.O_RDWR + posix_ipc.O_CREAT
        self._sem = posix_ipc.Semaphore(name, flags=flags, mode=0o600)
        self._name = name

    def lock(self, timeout=None):
        try:
            self._sem.acquire(timeout=timeout)
        except posix_ipc.BusyError:
            print("Can't lock semaphore " + self._name)
            return False
        return True

    def unlock(self):
        if self._sem.value == 0:
            self._sem.release()

    def __del__(self):
        self.unlock()