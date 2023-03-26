import time


class ElapsedTimePrinter:
    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.elapsed_time = time.time() - self.start_time
        self.print()

    def print(self):
        print(
            "elapsed time = {0:d}h {1:d}m {2:d}s".format(
                int(self.elapsed_time / 3600),
                int((self.elapsed_time % 3600) / 60),
                int((self.elapsed_time % 3600) % 60),
            )
        )
