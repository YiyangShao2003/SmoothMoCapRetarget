import joblib
import numpy as np
import mujoco
import time
import os
from mujoco.viewer import launch_passive


def mujoco_visualize(root_positions, root_orientations_quat, joint_positions):
    """
    Visualize motion data using MuJoCo.

    Parameters:
    - root_positions: np.ndarray of shape (T, 3), root positions over time
    - root_orientations_quat: np.ndarray of shape (T, 4), root orientations (quaternions) over time
    - joint_positions: np.ndarray of shape (T, n_joints), joint positions over time
    """
    # Path to your MuJoCo XML model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../resources/robots/g1/g1_27dof.xml')

    try:
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return

    T = root_positions.shape[0]

    # Confirm joint count matches
    floating_base_dof = 7  # x, y, z, quat_x, quat_y, quat_z, quat_w
    n_joints = model.nq - floating_base_dof  # Total DoFs minus floating base DoFs
    if joint_positions.shape[1] != n_joints:
        print(f"Joint positions dimension mismatch: expected {n_joints}, got {joint_positions.shape[1]}")
        return

    # Playback settings
    # Set fixed interval for 30Hz
    playback_speed = 1.0  # Real-time playback
    interval = 1.0 / 30.0  # Fixed interval for 30Hz

    try:
        # Launch the passive viewer
        with launch_passive(model, data) as viewer:
            print("Viewer launched successfully.")
            while viewer.is_running():  # Loop until the user closes the window
                for t in range(T):
                    if not viewer.is_running():
                        print("Viewer has been closed by the user.")
                        break

                    # Set root position and orientation
                    pos = root_positions[t]
                    quat = root_orientations_quat[t]

                    # Update qpos for the floating base
                    data.qpos[0:3] = pos
                    data.qpos[3:7] = quat

                    # Update joint positions
                    data.qpos[7:] = joint_positions[t]

                    # Optionally, zero velocities since we set positions directly
                    data.qvel[:] = 0

                    # Perform a forward pass to compute derived quantities
                    mujoco.mj_forward(model, data)

                    # Sync the viewer to update visualization
                    viewer.sync()

                    # Maintain fixed frame rate
                    time.sleep(interval)

    except Exception as e:
        print(f"An error occurred during visualization: {e}")
    finally:
        # Ensure that the viewer is properly closed
        print("Visualization ended.")


if __name__ == "__main__":
    try:
        # Path to your motion data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        motion_data_path = os.path.join(current_dir, "../data/g1/humanml3d_train_retargeted_wholebody_82.pkl")

        # Load the motion dataset
        motion_dataset = joblib.load(motion_data_path)
        
        # Remove the 'config' key if it exists
        motion_dataset.pop("config", None)
        
        start_index = 15
        
        # List of motion names to visualize
        l_original_motion_names = list(motion_dataset.keys())
        l_visualize_motion_names = []
        for motion_name in l_original_motion_names:
            if "M" in motion_name:
                continue
            motion_index = 0
            if motion_name[:-4] != "000000":
                motion_index = int(motion_name[:-4].lstrip('0'))
            if motion_index >= start_index:
                l_visualize_motion_names.append((motion_index, motion_name))
        l_visualize_motion_names.sort()
        l_visualize_motion_names = [x[1] for x in l_visualize_motion_names]
        
        # All joint names in the robot model
        robot_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        
        # Active joint names used for visualization
        active_robot_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        
        # Get indices of active joints in the robot model
        active_robot_joint_indices = []
        for joint_name in active_robot_joint_names:
            try:
                index = robot_joint_names.index(joint_name)
                active_robot_joint_indices.append(index)
            except ValueError:
                print(f"Joint name '{joint_name}' not found in the robot model.")
        
        # Iterate over each motion and visualize
        for motion_name in l_visualize_motion_names:
            # Check if motion_name exists in the dataset
            if motion_name not in motion_dataset:
                print(f"Motion '{motion_name}' not found in the dataset. Skipping.")
                continue
            
            motion_data = motion_dataset[motion_name]
            
            # Extract root position
            root_position = motion_data.get("root_trans_offset")
            if root_position is None:
                print(f"Root position data missing for motion '{motion_name}'. Skipping.")
                continue  # Skip if root position is not available
            
            # Extract root orientation (quaternions)
            root_orientation = motion_data.get("root_rot")
            if root_orientation is None:
                print(f"Root orientation data missing for motion '{motion_name}'. Skipping.")
                continue  # Skip if root orientation is not available
            
            # Convert quaternions from XYZW to WXYZ if necessary
            if root_orientation.shape[1] == 4:
                root_orientation = root_orientation[:, [3, 0, 1, 2]]
            else:
                print(f"Unexpected quaternion shape for motion '{motion_name}': {root_orientation.shape}. Skipping.")
                continue  # Skip if quaternion shape is incorrect
            
            # Extract joint positions
            joint_positions_all = motion_data.get("dof")
            if joint_positions_all is None:
                print(f"Joint positions data missing for motion '{motion_name}'. Skipping.")
                continue  # Skip if joint positions are not available
            
            if joint_positions_all.shape[1] < max(active_robot_joint_indices) + 1:
                print(f"Joint positions dimension mismatch for motion '{motion_name}'. Skipping.")
                continue  # Skip if joint positions are insufficient
            
            joint_positions = joint_positions_all[:, active_robot_joint_indices]  # Shape: (T, n_joints)
            
            print(f"Visualizing motion '{motion_name}'...")
            
            print(f"caption: {motion_data.get('captions')}")
            
            # Visualize the motion
            mujoco_visualize(root_position, root_orientation, joint_positions)
            
            print(f"Completed visualization for motion '{motion_name}'.\n")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")