import sys, time, os
import numpy as np

os.environ['CYCLONEDDS_URI'] = '<CycloneDDS><Domain><General><NetworkInterfaceAddress>wlp1s0</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
sys.path.append('/home/ntu_admin/xr_teleoperate')
sys.path.append('/home/ntu_admin/Isaac-GR00T')

from gr00t.policy.server_client import PolicyClient
from teleop.robot_control.robot_arm import G1_29_ArmController

policy = PolicyClient(host="localhost", port=5556)
arm_ctrl = G1_29_ArmController(simulation_mode=True)
time.sleep(5)

current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()

left_arm = current_lr_arm_q[:7].astype(np.float32)
right_arm = current_lr_arm_q[7:14].astype(np.float32)

robot_pos = np.zeros(3, dtype=np.float32)
left_leg = np.zeros(6, dtype=np.float32)
right_leg = np.zeros(6, dtype=np.float32)
waist = np.zeros(3, dtype=np.float32)
left_hand = np.zeros(7, dtype=np.float32)
right_hand = np.zeros(7, dtype=np.float32)

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
    'language': {
        'annotation.human.task_description': [["Pick up the red block"]]
    },
}

action, info = policy.get_action(obs)
print(f"SUCCESS! GR00T zero-shot pipeline working.")
print(f"Action type: {type(action)}")
print(f"Action keys: {action.keys() if isinstance(action, dict) else 'N/A'}")
if isinstance(action, dict):
    for key, val in action.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape {val.shape}, range [{val.min():.3f}, {val.max():.3f}]")
