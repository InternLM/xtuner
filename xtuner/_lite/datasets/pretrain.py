from xtuner._lite import get_logger
from .text import SoftPackerForText

logger = get_logger()


class SoftPackerForPretrain(SoftPackerForText):

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        if self._cached:
            self.load_cache()

        dataset = self.dataset
        pack_info = self.pack_info

        packed_items = pack_info[item]['indices']
        assert len(packed_items) > 0

        input_ids = []
        num_tokens = []
        for i in packed_items:
            input_ids.extend(dataset[i]['input_ids'])
            _num_tokens = dataset[i]['num_tokens']
            num_tokens.append(_num_tokens)

        if len(input_ids) < self.max_length:
            num_pad_tokens = self.max_length - len(input_ids)
            input_ids.extend([DEFAULT_PAD_TOKEN_INDEX] * num_pad_tokens)
            num_tokens.append(num_pad_tokens)
        else:
            num_tokens.append(0)

        packed = {
            'input_ids': input_ids,
            'labels': input_ids,
            'num_tokens': num_tokens,
        }

        if len(input_ids) != len(labels):
            logger.error(f'[packed_items] {packed_items}')
            logger.error(f'[input_ids] {input_ids}')
            logger.error(f'[labels] {labels}')
            logger.error(f'[num_tokens] {num_tokens}')
            raise RuntimeError('The lengths of input_ids and labels must be '
                               f'equal, but  found {len(input_ids)} and '
                               f'{len(labels)}.')

        if self.cached:
            self._free()

        return packed
