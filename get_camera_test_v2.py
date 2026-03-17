import sys
sys.path.append('/home/ntu_admin/xr_teleoperate')

from multiprocessing import shared_memory
import numpy as np

print("Checking for Isaac Sim camera shared memory...")

try:
    # Isaac Sim creates this shared memory
    shm = shared_memory.SharedMemory(name='isaac_multi_image_shm')
    print(f"SUCCESS! Found shared memory: {shm.name}, size: {shm.size} bytes")
    shm.close()
except FileNotFoundError:
    print("FAILED. Shared memory 'isaac_multi_image_shm' not found.")
    print("Isaac Sim may not be publishing camera images.")
