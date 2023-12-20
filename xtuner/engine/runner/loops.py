from mmengine.runner import EpochBasedTrainLoop as MMEngineEpochBasedTrainLoop


class EpochBasedTrainLoop(MMEngineEpochBasedTrainLoop):

    def _get_micro_batch(self, data_batch, idx):
        assert set(data_batch.keys()) == set(['data', 'data_samples'])
        data_batch_cur = {}
        data_batch_cur['data_samples'] = None
        data_cur = {}
        for key, val in data_batch['data'].items():
            if key == 'cumulative_len':
                data_cur[key] = [val[idx]]
            elif key == 'max_seqlen':
                data_cur[key] = val[idx]
            else:
                data_cur[key] = val[idx:idx+1]
        data_batch_cur['data'] = data_cur
        return data_batch_cur
    
    def run_iter(self, idx, data_batch) -> None:
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        
        for i in range(4):
            outputs = self.runner.model.train_step(
                self._get_micro_batch(data_batch, i), optim_wrapper=self.runner.optim_wrapper)
        
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
