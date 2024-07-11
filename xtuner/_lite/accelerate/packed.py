from contextlib import contextmanager


@contextmanager
def packed_sequence(num_tokens, enable=False):
    from mmengine import MessageHub
    ctx = MessageHub.get_instance('packed_sequence')
    
    if enable:
        ctx.update_info('num_tokens', num_tokens)
        # ctx.update_info('image_ranges', image_ranges)
        
    else:
        ctx.update_info('num_tokens', None)
        # ctx.update_info('image_ranges', None)

    yield
    
    ctx.update_info('num_tokens', None)
    # ctx.update_info('image_ranges', None)
