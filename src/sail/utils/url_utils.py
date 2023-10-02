import socket
from urllib.parse import urlparse


def validate_host_and_port(host, port, throw_exception=True):
    validation_success = True
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        if result != 0:
            validation_success = False
            if throw_exception:
                raise Exception(f"Port: {port} is not open on the host: {host}")

    except socket.gaierror as e:
        validation_success = False
        if throw_exception:
            raise Exception(f"{host}: {str(e)}")
    finally:
        sock.close()

    return validation_success


def validate_address(address, throw_exception=True):
    endpoint = urlparse(address)
    return validate_host_and_port(endpoint.hostname, endpoint.port, throw_exception)
