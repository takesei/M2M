import RPi.GPIO as GPIO
import time

# Pin Definitions
pos = 11
neg = 13
rot = 15

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup((pos, neg, rot), GPIO.OUT)

    for i in range(3):
        GPIO.output(pos, GPIO.HIGH)
        time.sleep(0.22)
        GPIO.output(pos, GPIO.LOW)
        time.sleep(1)
    GPIO.cleanup()

if __name__ == '__main__':
    GPIO.cleanup()
    main()
