import socket


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.254.254.254', 1))
        local_ip = s.getsockname()[0]
    except BaseException:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip


def get_ip_hostname():
    hostname = socket.gethostname()
    return get_ip(), hostname


def get_free_port() -> int:
    """Get a free port for the actor to use for DDP dist_init.

    Returns: A free port that could be used.
    """
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port
