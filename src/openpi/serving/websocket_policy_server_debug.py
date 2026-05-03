import asyncio
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _to_msgpack_safe(x: Any) -> Any:
    """Recursively convert leaves to msgpack-friendly values.

    msgpack_numpy can handle NumPy arrays, but this also handles JAX arrays,
    NumPy scalars, nested tuples/lists, and dictionaries in debug payloads.
    """
    if isinstance(x, dict):
        return {k: _to_msgpack_safe(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [_to_msgpack_safe(v) for v in x]
    if isinstance(x, list):
        return [_to_msgpack_safe(v) for v in x]
    if isinstance(x, (str, bytes, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    try:
        return np.asarray(x)
    except Exception:
        return x


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_start = time.monotonic()
                policy_output = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                # Preferred new format from policy.py:
                #   {"actions": ..., "debug": {...}, ...}
                # Backward-compatible old format:
                #   raw actions array
                if isinstance(policy_output, dict):
                    response = dict(policy_output)
                else:
                    response = {"actions": policy_output}

                response["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                response = _to_msgpack_safe(response)
                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
