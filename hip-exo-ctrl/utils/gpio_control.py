# utils/gpio_control.py
import Jetson.GPIO as GPIO


class GPIOControl:
    """Manages a single digital output pin for sync pulse generation."""

    def __init__(self, pin: int):
        self.pin = int(pin)
        self._state = 0
        try:
            GPIO.setwarnings(False)
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        print(f"GPIO initialized on pin {self.pin}")

    def pulse_start(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self._state = 1

    def pulse_end(self):
        GPIO.output(self.pin, GPIO.LOW)
        self._state = 0

    def state(self) -> int:
        return self._state

    def close(self):
        try:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup()
            print("GPIO cleaned up")
        except Exception:
            pass
