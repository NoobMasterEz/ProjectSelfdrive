from UI.App import main
from subprocess import call
import threading

def process1():
    call(["python", "Autopilot-TensorFlow/run.py"])

def process2():
    main()

if __name__ == "__main__":
    taget1 = threading.Thread(target=process1)
    taget1.start()
    taget2 = threading.Thread(target=process2)
    taget2.start()

