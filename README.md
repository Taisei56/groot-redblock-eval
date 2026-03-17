# GR00T N1.6 Fine-tuning for Dexterous Manipulation

Fine-tuning NVIDIA's GR00T N1.6-3B Vision-Language-Action model on a Unitree G1 humanoid robot with Dex3-1 dexterous hands for a red block pick-and-place task.

This is part of an undergraduate research project (URECA) at the Schaeffler-NTU Corporate Lab, NTU Singapore.

## Overview

The goal is to fine-tune a pretrained VLA model on a custom robot embodiment using teleoperated demonstrations, then evaluate it in simulation. The full pipeline covers data collection via Apple Vision Pro teleoperation, dataset conversion, model fine-tuning, and closed-loop evaluation in Isaac Sim.

Robot: Unitree G1 (29 DOF arms) with Dex3-1 hands (7 DOF each)  
Task: Pick up a red block and place it on a yellow square  
Simulator: NVIDIA Isaac Sim 5.0 via IsaacLab  
Model: GR00T N1.6-3B (Cosmos-Reason-2B VLM + DiT action head)

## Results

Open-loop evaluation on trajectory 0 (53 demonstrations, 4000 training steps):

| | MSE | MAE |
|---|---|---|
| Base model (zero-shot) | 0.433 | 0.458 |
| Fine-tuned | 0.00077 | 0.0164 |

562x reduction in MSE. The fine-tuned model produces smooth, structured motion that closely tracks demonstration trajectories. The base model produces random oscillation with no task-relevant structure.

## Pipeline

Data collection: Apple Vision Pro teleoperation via xr_teleoperate, recording 53 demonstrations at 30 Hz across 3 cameras (head, left wrist, right wrist) and 28 DOF joint states.

Dataset conversion: Custom robot config `G1_DEX3_XR` added to unitree_lerobot for converting raw JSON teleoperation data to LeRobot format. Videos re-encoded to H.264 for torchcodec compatibility.

Modality config: Custom `new_embodiment` config registered in GR00T's embodiment system, mapping state/action to left_arm, right_arm, left_ee, right_ee with appropriate action representations (absolute, non-eef).

Processor injection: Dataset statistics (mean/std per joint group) injected into the GR00T processor for correct normalization during both training and inference.

Fine-tuning: 4000 steps, global batch size 64, learning rate 1e-4 with cosine schedule, DeepSpeed ZeRO Stage 2 across 2x RTX 6000 Ada GPUs. Loss converged from 1.34 to 0.027.

Evaluation: Closed-loop evaluation in Isaac Sim using DDS communication for robot state/action, shared memory for camera images, and ZMQ for policy server communication.

## Key Files

`zero_shot_eval_redblock.py` — closed-loop evaluation script with action chunking, hand execution, video recording, and scene reset  
`gr00t_new_embodiment_processor/` — custom processor with injected modality config and dataset statistics

## Stack

Isaac Sim 5.0, IsaacLab, ROS2, GR00T N1.6, LeRobot, unitree_sdk2py, DDS, Apple Vision Pro, Python, PyTorch, DeepSpeed, HuggingFace Transformers

## Status

Fine-tuning complete. Closed-loop sim evaluation shows grasping behavior with smooth arm trajectories. Ongoing work: full pick-and-place task completion, real robot deployment.
