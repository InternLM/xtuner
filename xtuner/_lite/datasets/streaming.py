# Copyright (c) OpenMMLab. All rights reserved.


class Streaming:
    def __init__(self, file, max_epoch=1):
        self.file = file
        self.offset = 0
        self.epoch = 1
        self.max_epoch = max_epoch

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.file) as f:
            f.seek(self.offset)
            line = f.readline()

            if not line and self.epoch < self.max_epoch:
                self.offset = 0
                self.epoch += 1
                return next(self)

            elif not line and self.epoch == self.max_epoch:
                raise StopIteration

            self.offset = f.tell()
        return line
