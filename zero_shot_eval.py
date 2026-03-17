import sys, time, os
import numpy as np
from multiprocessing import shared_memory

os.environ['CYCLONEDDS_URI'] = '<CycloneDDS><Domain><General><NetworkInterfaceAddress>wlp1s0</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
sys.path.append('/home/ntu_admin/xr_teleoperate')
sys.path.append('/home/ntu_admin/Isaac-GR00T')

from gr00t.policy.server_client import PolicyClient
from teleop.robot_control.robot_arm import G1_29_ArmController

policy = PolicyClient(host="localhost", port=5556)
arm_ctrl = G1_29_ArmController(simulation_mode=True)
shm = shared_memory.SharedMemory(name='isaac_multi_image_shm')
img_array = np.ndarray((3, 480, 640, 3), dtype=np.uint8, buffer=shm.buf)

time.sleep(5)

num_episodes = 10
max_steps = 200

print(f"Starting zero-shot evaluation: {num_episodes} episodes")

for ep in range(num_episodes):
    print(f"\nEpisode {ep+1}/{num_episodes}")
    input("Press Enter to start episode (reset robot manually)...")
    
    for step in range(max_steps):
        # Get state
        current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
        left_arm = current_lr_arm_q[:7].astype(np.float32)
        right_arm = current_lr_arm_q[7:14].astype(np.float32)
        
        # Dummy body parts
        robot_pos = np.zeros(3, dtype=np.float32)
        left_leg = np.zeros(6, dtype=np.float32)
        right_leg = np.zeros(6, dtype=np.float32)
        waist = np.zeros(3, dtype=np.float32)
        left_hand = np.zeros(7, dtype=np.float32)
        right_hand = np.zeros(7, dtype=np.float32)
        
        # Real camera
        real_img = img_array[0][np.newaxis, np.newaxis, :, :, :]
        
        obs = {
            'video': {'ego_view': real_img},
            'state': {
                'robot_pos': robot_pos[np.newaxis, np.newaxis, :],
                'left_leg': left_leg[np.newaxis, np.newaxis, :],
                'right_leg': right_leg[np.newaxis, np.newaxis, :],
                'waist': waist[np.newaxis, np.newaxis, :],
                'left_arm': left_arm[np.newaxis, np.newaxis, :],
                'right_arm': right_arm[np.newaxis, np.newaxis, :],
                'left_hand': left_hand[np.newaxis, np.newaxis, :],
                'right_hand': right_hand[np.newaxis, np.newaxis, :],
            },
            'language': {'annotation.human.task_description': [["Pick up the red block"]]},
        }
        
        # Get action from GR00T
        action, info = policy.get_action(obs)
        
        # TODO: Send action to robot via DDS
        # For now, just print progress
        if step % 50 == 0:
            print(f"  Step {step}/{max_steps}")
        
        time.sleep(0.033)  # 30 Hz
    
    success = input("Was episode successful? (y/n): ").lower() == 'y'
    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")

shm.close()
print("\nEvaluation complete.")
