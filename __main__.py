# -*- coding: utf-8 -*-
# start mitmproxy server
import os
import signal
import subprocess

try:
    p = subprocess.check_call(["mitmdump", "-s", "addons.py"],
                              cwd=os.path.dirname(__file__))
except KeyboardInterrupt:
    p.send_signal(signal.SIGINT)
