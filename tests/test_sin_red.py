import socket

import pytest


def test_los_tests_no_pueden_conectarse_fuera_de_la_maquina():
    # Dirección IP directa: no depende del DNS
    with socket.socket() as sock, pytest.raises(RuntimeError, match="Conexión externa bloqueada"):
        sock.connect(("8.8.8.8", 443))
