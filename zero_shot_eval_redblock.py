import sys
import time
import os
import threading
import numpy as np
import cv2

os.environ['CYCLONEDDS_URI'] = '<CycloneDDS><Domain><General><NetworkInterfaceAddress>wlp1s0</NetworkInterfaceAddress></General></Domain></CycloneDDS>'

sys.path.append('/home/ntu_admin/xr_teleoperate')
sys.path.append('/home/ntu_admin/Isaac-GR00T')
sys.path.append('/home/ntu_admin/unitree_sim_isaaclab')

from gr00t.policy.server_client import PolicyClient
from teleop.robot_control.robot_arm import G1_29_ArmController
from image_server.shared_memory_utils import MultiImageReader

# Note: ChannelFactoryInitialize is NOT called here.
# G1_29_ArmController(simulation_mode=True) calls ChannelFactoryInitialize(1)
# which is the correct domain for Isaac Sim. All subscribers must be created
# after G1_29_ArmController is initialised so they share domain 1.
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState, HandState_ as hg_HandState, HandCmd_ as hg_HandCmd
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


# Dex3 DDS topic names
DEX3_LEFT_STATE_TOPIC  = "rt/dex3/left/state"
DEX3_RIGHT_STATE_TOPIC = "rt/dex3/right/state"
DEX3_NUM_MOTORS = 7

# GR00T UNITREE_G1 embodiment expects these state keys
# based on what we discovered during the previous zero shot test
TASK_DESCRIPTION = "Pick up the red block and place it on the yellow square"

# Target frequency
CONTROL_HZ = 30
CONTROL_DT  = 1.0 / CONTROL_HZ


class Dex3StateReader:
    """
    Subscribes to Dex3 left and right hand DDS state topics.
    Stores the latest joint positions in thread-safe arrays.
    """
    def __init__(self):
        self.left_qpos  = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
        self.right_qpos = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
        self._lock = threading.Lock()
        self._left_received  = False
        self._right_received = False

        self._left_sub = ChannelSubscriber(DEX3_LEFT_STATE_TOPIC, hg_HandState)
        self._left_sub.Init(self._left_callback, 10)

        self._right_sub = ChannelSubscriber(DEX3_RIGHT_STATE_TOPIC, hg_HandState)
        self._right_sub.Init(self._right_callback, 10)

        print("[Dex3StateReader] Subscribed to Dex3 left and right state topics")

    def _left_callback(self, msg):
        with self._lock:
            for i in range(DEX3_NUM_MOTORS):
                self.left_qpos[i] = msg.motor_state[i].q
            self._left_received = True

    def _right_callback(self, msg):
        with self._lock:
            for i in range(DEX3_NUM_MOTORS):
                self.right_qpos[i] = msg.motor_state[i].q
            self._right_received = True

    def get_state(self):
        with self._lock:
            return self.left_qpos.copy(), self.right_qpos.copy()

    def is_ready(self):
        return self._left_received and self._right_received


class CameraReader:
    """
    Reads camera images from Isaac Sim shared memory.
    Isaac Sim writes head, left, right camera frames concatenated
    horizontally into shared memory named isaac_multi_image_shm.
    MultiImageReader splits them back into individual frames.
    Images come in BGR format from OpenCV — convert to RGB for GR00T.
    """
    def __init__(self):
        self.reader = MultiImageReader()
        self._last_images = None
        print("[CameraReader] Shared memory reader initialised")

    def get_images(self):
        images = self.reader.read_images()
        if images is not None:
            self._last_images = images
        return self._last_images

    def is_ready(self):
        images = self.reader.read_images()
        if images is not None:
            self._last_images = images
            return True
        return False


class SceneResetter:
    """
    Publishes reset commands to Isaac Sim via DDS topic rt/reset_pose/cmd.
    reset_category '1' resets the object (red block) position only.
    reset_category '2' resets everything including the robot.
    Must be initialised after G1_29_ArmController so DDS domain 1 is active.
    """
    def __init__(self):
        self._pub = ChannelPublisher("rt/reset_pose/cmd", String_)
        self._pub.Init()
        print("[SceneResetter] Reset publisher initialised")

    def reset_object(self):
        msg = String_(data="1")
        self._pub.Write(msg)
        time.sleep(0.5)
        print("[SceneResetter] Object reset sent")

    def reset_all(self):
        msg = String_(data="2")
        self._pub.Write(msg)
        time.sleep(1.0)
        print("[SceneResetter] Full reset sent")


def prepare_observation(arm_ctrl, dex3_reader, camera_reader):
    """
    Build observation dict for new_embodiment modality config.
    State keys: left_arm(7), right_arm(7), left_ee(7), right_ee(7)
    Video keys: head, left_wrist, right_wrist
    """
    # arm state
    lr_arm_q = arm_ctrl.get_current_dual_arm_q()
    left_arm  = lr_arm_q[:7].astype(np.float32)
    right_arm = lr_arm_q[7:14].astype(np.float32)

    # hand state from Dex3
    left_ee, right_ee = dex3_reader.get_state()
    left_ee  = left_ee.astype(np.float32)
    right_ee = right_ee.astype(np.float32)

    def add_dims(arr):
        return arr[np.newaxis, np.newaxis, :]

    state = {
        'left_arm':  add_dims(left_arm),
        'right_arm': add_dims(right_arm),
        'left_ee':   add_dims(left_ee),
        'right_ee':  add_dims(right_ee),
    }

    # cameras
    images = camera_reader.get_images()

    def get_camera(key):
        if images is not None and key in images:
            bgr = images[key]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (480, 640):
                rgb = cv2.resize(rgb, (640, 480))
            return rgb
        else:
            print(f"[WARNING] No {key} camera image, using black frame")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    video = {
        'head':        get_camera('head')[np.newaxis, np.newaxis, :],
        'left_wrist':  get_camera('left')[np.newaxis, np.newaxis, :],
        'right_wrist': get_camera('right')[np.newaxis, np.newaxis, :],
    }

    language = {
        'annotation.human.task_description': [[TASK_DESCRIPTION]]
    }

    return {'video': video, 'state': state, 'language': language}


# Global hand publishers — initialized once
_left_hand_pub  = None
_right_hand_pub = None

def init_hand_publishers():
    global _left_hand_pub, _right_hand_pub
    _left_hand_pub  = ChannelPublisher("rt/dex3/left/cmd",  hg_HandCmd)
    _right_hand_pub = ChannelPublisher("rt/dex3/right/cmd", hg_HandCmd)
    _left_hand_pub.Init()
    _right_hand_pub.Init()
    print("[HandPublisher] Left and right hand publishers initialized")

def send_hand_cmd(publisher, q_target):
    """Send position command to one hand via DDS."""
    msg = unitree_hg_msg_dds__HandCmd_()
    for i in range(min(7, len(q_target))):
        msg.motor_cmd[i].q   = float(q_target[i])
        msg.motor_cmd[i].kp  = 1.0
        msg.motor_cmd[i].kd  = 0.1
        msg.motor_cmd[i].dq  = 0.0
        msg.motor_cmd[i].tau = 0.0
    publisher.Write(msg)

def execute_action_step(action, step_idx, arm_ctrl, dex3_reader):
    """
    Execute a single timestep from the action chunk.
    action keys: left_arm(1,16,7), right_arm(1,16,7), left_ee(1,16,7), right_ee(1,16,7)
    """
    i = min(step_idx, action['left_arm'].shape[1] - 1)
    # arm commands
    left_arm_action  = action['left_arm'][0, i, :].astype(np.float64)
    right_arm_action = action['right_arm'][0, i, :].astype(np.float64)
    combined_arm     = np.concatenate([left_arm_action, right_arm_action])
    arm_ctrl.ctrl_dual_arm(combined_arm, np.zeros(14))
    # hand commands
    if _left_hand_pub is not None and 'left_ee' in action:
        left_ee_action = action['left_ee'][0, i, :]
        send_hand_cmd(_left_hand_pub, left_ee_action)
    if _right_hand_pub is not None and 'right_ee' in action:
        right_ee_action = action['right_ee'][0, i, :]
        send_hand_cmd(_right_hand_pub, right_ee_action)

def wait_for_ready(arm_ctrl, dex3_reader, camera_reader, timeout=30):
    """Wait until all data sources are providing data."""
    print("Waiting for Isaac Sim data sources to become ready...")
    start = time.time()
    while time.time() - start < timeout:
        arm_ready    = True
        dex3_ready   = dex3_reader.is_ready()
        camera_ready = camera_reader.is_ready()

        status = (
            f"  Arms: {'OK' if arm_ready else 'waiting'} | "
            f"Dex3: {'OK' if dex3_ready else 'waiting'} | "
            f"Camera: {'OK' if camera_ready else 'waiting'}"
        )
        print(status, end='\r')

        if arm_ready and dex3_ready and camera_ready:
            print("\nAll data sources ready.")
            return True
        time.sleep(0.5)

    print(f"\nTimeout after {timeout}s. Proceeding anyway.")
    return False


def main():
    print("Initialising GR00T zero shot evaluation")
    print(f"Task: {TASK_DESCRIPTION}")
    print()

    # connect to GR00T policy server first (no DDS needed)
    print("Connecting to GR00T policy server on localhost:5556...")
    policy = PolicyClient(host="localhost", port=5556)
    if not policy.ping():
        print("ERROR: Cannot connect to policy server.")
        print("Make sure Terminal 1 (GR00T server) is running first.")
        return
    print("Policy server connected.")

    # initialise arm controller FIRST — this calls ChannelFactoryInitialize(1)
    # for simulation domain. All DDS subscribers must come after this.
    print("Initialising arm controller (this sets up DDS domain 1)...")
    arm_ctrl = G1_29_ArmController(simulation_mode=True)

    # initialise Dex3 state reader AFTER arm controller
    print("Initialising hand publishers...")
    init_hand_publishers()
    print("Initialising Dex3 state reader...")
    dex3_reader = Dex3StateReader()

    # initialise scene resetter AFTER arm controller
    print("Initialising scene resetter...")
    resetter = SceneResetter()

    # initialise camera reader
    print("Initialising camera reader from shared memory...")
    camera_reader = CameraReader()

    # wait for all sources to be live
    wait_for_ready(arm_ctrl, dex3_reader, camera_reader, timeout=30)

    # evaluation parameters
    num_episodes  = 5
    max_steps     = 500
    results       = []

    print(f"\nStarting evaluation: {num_episodes} episodes, {max_steps} steps each")
    print(f"Control frequency: {CONTROL_HZ} Hz")
    print()

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")
        print("  Resetting scene...")
        arm_ctrl.ctrl_dual_arm_go_home()
        time.sleep(2.0)
        resetter.reset_all()
        time.sleep(1.0)
        print("  Scene reset. Starting episode...")

        step_times = []

        action_chunk = None
        chunk_step = 0
        ACTION_HORIZON = 16
        for step in range(max_steps):
            t_start = time.time()

            if action_chunk is None or chunk_step >= ACTION_HORIZON:
                obs = prepare_observation(arm_ctrl, dex3_reader, camera_reader)
                action_chunk, info = policy.get_action(obs)
                chunk_step = 0
            execute_action_step(action_chunk, chunk_step, arm_ctrl, dex3_reader)
            chunk_step += 1

            # sleep to maintain 30Hz then measure full loop time
            time.sleep(max(0, CONTROL_DT - (time.time() - t_start)))
            step_times.append(time.time() - t_start)


            if step % 50 == 0:
                avg_hz = 1.0 / np.mean(step_times[-50:]) if step_times else 0
                print(f"  Step {step:3d}/{max_steps} | avg freq: {avg_hz:.1f} Hz")

        success_input = input("  Episode complete. Success? (y/n): ").strip().lower()
        success = success_input == 'y'
        results.append(success)
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
        print()

    # summary
    num_success = sum(results)
    success_rate = num_success / num_episodes * 100
    print("Evaluation complete")
    print(f"Results: {num_success}/{num_episodes} = {success_rate:.1f}%")
    for i, r in enumerate(results):
        print(f"  Episode {i + 1}: {'SUCCESS' if r else 'FAILED'}")


if __name__ == "__main__":
    main()
