import Jetson.GPIO as GPIO
import time

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


class Conveyer():
    """
    Control GPIO aounrd Belt Conveyer

    Attributes
    ----------
    pos: int
      Positive Pin num connected to MotorDriver
    neg: int
      Negative Pin num connected to MotorDriver
    conv: int
      Pin num connected to Conveyer
    mode: int=GPIO.BOARD
      Num indicating method for GPIO

    """

    def __init__(self, pos, neg, conv, nconv, mode=GPIO.BOARD):
        GPIO.setmode(mode)
        GPIO.setup((pos, neg, conv, nconv), GPIO.OUT)
        self.pos = pos
        self.neg = neg
        self.conv = conv
        self.nconv = nconv
        logger.debug(f"GPIO module initialized as pos:{pos}, neg:{neg}, conv:{conv}, nconv:{nconv} @mode: {mode}")

    def drop_out(self, sec=0.5):
        logger.debug("Stuff staged off")
        GPIO.output(self.pos, GPIO.HIGH)
        time.sleep(sec)
        GPIO.output(self.pos, GPIO.LOW)
        GPIO.output(self.neg, GPIO.HIGH)
        time.sleep(sec)
        GPIO.output(self.neg, GPIO.LOW)
        return self

    def convey(self, position, sec=0.75):
        logger.debug(f"Convey Stuff for {position}")
        assert 1 <= position <= 3, "OUT OF RANGE: range of pos is [0,2]"
        GPIO.output(self.conv, GPIO.HIGH)
        time.sleep(sec*(position))
        GPIO.output(self.conv, GPIO.LOW)
        time.sleep(0.5)
        return self

    def __del__(self):
        GPIO.cleanup()


if __name__ == '__main__':
    # Pin Definitions
    pos = 11
    neg = 13
    conv = 15
    nconv = 19

    obj = Conveyer(pos, neg, conv, nconv)
    obj.convey(2).drop_out()
    del obj
