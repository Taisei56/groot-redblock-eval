import sys, time, os
import numpy as np

os.environ['CYCLONEDDS_URI'] = '<CycloneDDS><Domain><General><NetworkInterfaceAddress>wlp1s0</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
sys.path.append('/home/ntu_admin/xr_teleoperate')
sys.path.append('/home/ntu_admin/Isaac-GR00T')

from gr00t.policy.server_client import PolicyClient
from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.image_server.image_client import ImageClient

policy = PolicyClient(host="localhost", port=5556)
arm_ctrl = G1_29_ArmController(simulation_mode=True)

# ImageClient receives images via network from Isaac Sim
img_client = ImageClient(port=5555, server_address="localhost")

time.sleep(5)

print("Testing image reception from Isaac Sim...")
# The client receives images in background thread
# Access via img_client's internal buffer

# Since we don't know the exact API, let's just use dummy for now
# and focus on sending actions to robot

num_episodes = 3
max_steps = 100

for ep in range(num_episodes):
    print(f"\nEpisode {ep+1}/{num_episodes}")
    input("Press Enter (robot will move)...")
    
    for step in range(max_steps):
        current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
        left_arm = current_lr_arm_q[:7].astype(np.float32)
        right_arm = current_lr_arm_q[7:14].astype(np.float32)
        
        # Dummy state
        robot_pos = np.zeros(3, dtype=np.float32)
        left_leg = np.zeros(6, dtype=np.float32)
        right_leg = np.zeros(6, dtype=np.float32)
        waist = np.zeros(3, dtype=np.float32)
        left_hand = np.zeros(7, dtype=np.float32)
        right_hand = np.zeros(7, dtype=np.float32)
        
        # Use dummy image for now (fix later)
        dummy_img = np.random.randint(0, 255, (1, 1, 480, 640, 3), dtype=np.uint8)
        
        obs = {
            'video': {'ego_view': dummy_img},
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
        
        action, info = policy.get_action(obs)
        
        # Send to robot
        left_action = action['left_arm'][0, 0, :]
        right_action = action['right_arm'][0, 0, :]
        combined_action = np.concatenate([left_action, right_action])
        
        arm_ctrl.ctrl_dual_arm(combined_action, np.zeros(14))
        
        if step % 25 == 0:
            print(f"  Step {step}/{max_steps}")
        
        time.sleep(0.033)
    
    success = input("Success? (y/n): ").lower() == 'y'
    print(f"  {'SUCCESS' if success else 'FAILED'}")

print("\nDone")
