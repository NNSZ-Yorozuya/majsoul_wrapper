# -*- coding: utf-8 -*-
# 监听websocket，通过xmlrpc为其他程序提供抓包服务
import os
import pickle
import threading
from multiprocessing import Lock
from xmlrpc.server import SimpleXMLRPCServer

import mitmproxy.http
import mitmproxy.websocket

buffer = []
mutex = Lock()


class ClientWebSocket:
    def websocket_message(self, flow: mitmproxy.http.HTTPFlow):
        """

            Called when a WebSocket message is received from the client or

            server. The most recent message will be flow.messages[-1]. The

            message is user-modifiable. Currently there are two types of

            messages, corresponding to the BINARY and TEXT frame types.

        """
        flow_msg = flow.websocket.messages[-1]

        with mutex:
            buffer.append(flow_msg)

        packet = flow_msg.content
        from_client = flow_msg.from_client


addons = [
    ClientWebSocket()
]


# RPC调用函数
def take():
    global buffer

    with mutex:
        msgs = list(buffer)
        buffer.clear()

    return pickle.dumps(msgs)


def hello():
    return "hello"


def RPC_init():
    host = os.getenv("RPC_HOST") or '127.0.0.1'
    port = os.getenv("RPC_PORT") or 37247

    server = SimpleXMLRPCServer((host, port), logRequests=False)
    server.register_function(take, "take")
    server.register_function(hello, "hello")
    server.serve_forever()

    print(f"RPC Server Listening on {host}:{port} for Client.")


RPC_server = threading.Thread(target=RPC_init)
RPC_server.start()
