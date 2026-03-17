import sys, time
sys.path.append('/home/ntu_admin/xr_teleoperate')

from teleop.image_server.image_client import ImageClient

print("Testing camera image reception...")
img_client = ImageClient(port=5555)
time.sleep(2)

if img_client.shared_image is not None:
    print(f"SUCCESS! Camera image shape: {img_client.shared_image.shape}")
else:
    print("FAILED. No camera images received.")
    print("Check if Isaac Sim is running with --enable_cameras")
