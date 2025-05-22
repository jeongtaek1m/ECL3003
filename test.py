from base_ctrl import BaseController
import time, math, threading


base = BaseController('/dev/ttyUSB0', 115200)

def send_control(L: float, R: float):
    """BaseController 인터페이스 (로버는 -값이 전진)"""
    base.base_json_ctrl({"T": 1, "L": L, "R": R})


while (True):
    send_control(0.2,-0.2)
    time.sleep(5)
    send_control(-0.2,0.2)
    time.sleep(5)
