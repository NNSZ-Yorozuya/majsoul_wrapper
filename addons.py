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

    def __init__(self):
        pass

    # Websocket lifecycle

    def websocket_handshake(self, flow: mitmproxy.http.HTTPFlow):
        """

            Called when a client wants to establish a WebSocket connection. The

            WebSocket-specific headers can be manipulated to alter the

            handshake. The flow object is guaranteed to have a non-None request

            attribute.

        """
        print('[handshake websocket]:', flow, flow.__dict__, dir(flow))

    def websocket_start(self, flow: mitmproxy.http.HTTPFlow):
        """

            A websocket connection has commenced.

        """
        print('[new websocket]:', flow, flow.__dict__, dir(flow))

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

        # This is cheating, extending the time limit to 7 seconds
        # tamperUsetime(flow_msg)
        # result = liqi.parse(flow_msg)
        # print(result)
        # print('-'*65)

        packet = flow_msg.content
        from_client = flow_msg.from_client
        print("[" + ("Sended" if from_client else "Reveived") +
              "] from '" + flow.id + "': decode the packet here: %r…" % packet)

    def websocket_error(self, flow: mitmproxy.http.HTTPFlow):
        """

            A websocket connection has had an error.

        """

        print("websocket_error, %r" % flow)

    def websocket_end(self, flow: mitmproxy.http.HTTPFlow):
        """

            A websocket connection has ended.

        """
        print('[end websocket]:', flow, flow.__dict__, dir(flow))


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


def RPC_init():
    host = os.getenv("RPC_HOST") or '127.0.0.1'
    port = os.getenv("RPC_PORT") or 37247

    server = SimpleXMLRPCServer((host, port))
    server.register_function(take, "take")
    server.serve_forever()

    print(f"RPC Server Listening on {host}:{port} for Client.")


RPC_server = threading.Thread(target=RPC_init)
RPC_server.start()
