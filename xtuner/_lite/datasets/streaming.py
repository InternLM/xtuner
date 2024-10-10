import json

import numpy as np
from torch.utils.data import IterableDataset


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


# import torch

# class MultiStreamingDataset(torch.utils.data.IterableDataset):

#     def __init__(self, streamings, weights, max_length, tokenize_fn, seed, dp_rank, dp_world_size, crossover = False):

#         assert len(streamings) == len(weights)
#         self.streamings = streamings
#         self.activated = [True for _ in self.streamings]
#         for sid, stream in enumerate(self.streamings):
#             stream.offset = 0
#             try:
#                 for _ in range(self.dp_rank):
#                     next(stream)
#             except StopIteration:
#                 self.activated[sid] = False

#         self.random_state = np.random.RandomState(seed + dp_rank)
#         self.weights = weights

#         self.max_length = max_length
#         self.tokenize_fn = tokenize_fn
#         self.dp_rank = dp_rank
#         self.dp_world_size = dp_world_size
#         self.crossover = crossover

#     def reactivate(self):
#         self.activated = [True for _ in self.streamings]
#         for stream in self.streamings:
#             stream.offset = 0
#             for _ in range(self.dp_rank):
#                 next(stream)

#     @property
#     def probabilities(self):
#         if sum(self.activated) == 0:
#             self.reactivate()

#         probs = (np.array(self.weights) * self.activated) / sum(self.weights[self.activated])
#         return probs

#     @property
#     def num_streamings(self):
#         assert len(self.iterators) == len(self.weights)
#         return len(self.weights)

#     def per_rank_next(self, streaming_id):

#         sid = streaming_id
#         streaming = self.streamings[sid]

#         try:
#             data = next(streaming)
#         except StopIteration:
#             self.activated[sid] = False
#             sid = self.random_state.choice(
#                         self.num_streamings, p=self.probabilities)
#             return self.per_rank_next(sid)

#         try:
#             for _ in range(self.dp_world_size):
#                 next(streaming)
#         except StopIteration:
#             self.activated[sid] = False

#         return data, sid

#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()

#         if worker_info and worker_info.num_workers > 1:
#             raise NotImplementedError

#         input_ids = []
#         labels = []
#         num_tokens = []
#         while True:
#             sid = self.random_state.choice(
#                 self.num_streamings, p=self.probabilities)

#             while len(input_ids) < self.max_length:
#                 if self.crossover:
#                     sid = self.random_state.choice(
#                         self.num_streamings, p=self.probabilities)

#                 line, sid = self.per_rank_next(sid)

#                 tokenized = self.tokenize_fn(json.loads(line))

#                 input_ids.extend(tokenized['input_ids'])
#                 labels.extend(tokenized['labels'])
#                 num_tokens.extend(tokenized['num_tokens'])

#             remain_tokens = max(sum(num_tokens) - self.max_length, 0)
#             num_tokens[-1] = num_tokens[-1] - remain_tokens

#             packed_ids = input_ids[:self.max_length]
#             packed_labels = labels[:self.max_length]
#             packed_tokens = num_tokens

#             if remain_tokens:
#                 input_ids = input_ids[self.max_length:]
#                 labels = labels[self.max_length:]
#                 num_tokens = [remain_tokens]

#             yield {'input_ids': packed_ids, 'labels': packed_labels, 'num_tokens': packed_tokens}

if __name__ == '__main__':
    import json
    streaming = Streaming(
        '/mnt/hwfile/xtuner/huanghaian/data/databricks-dolly-15k/databricks-dolly-15k.jsonl'
    )

    data = next(streaming)
    print(json.loads(data))

    data = next(streaming)
    print(json.loads(data))

    data = next(streaming)
    print(json.loads(data))

    data = next(streaming)
    print(json.loads(data))
