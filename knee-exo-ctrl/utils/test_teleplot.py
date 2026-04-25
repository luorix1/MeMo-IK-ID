import math
import sys
import time
sys.path.insert(0, "/home/mmlkneejetson/Documents/Rajiv/Jetson_Teensy_Comms")
from teleplot import Teleplot

tp = Teleplot("127.0.0.1", 47269)

print("Teleplot Test Sending... (Check VS Code)")

for i in range(1000):
    val = math.sin(i * 0.1)
    tp.sendValue("test_sin", val)
    print(f"Sending {i}: {val:.2f}") 
    
    time.sleep(0.05)

print("Done!")