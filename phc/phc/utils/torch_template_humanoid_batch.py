from platform import node
import re
from threading import local
from turtle import position
import torch
import numpy as np
import phc.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import poselib.poselib.core.rotation3d as pRot

TEMPLATE_ROTATION_AXIS = torch.tensor([[
    [1, 0, 0], # template x axis
    [0, 1, 0], # template y axis
    [0, 0, 1], # template z axis
]]
)

class Humanoid_Batch:
    
    def __init__(self, mjcf_file = f"path/to/robot.xml", device = torch.device("cpu")):
        self.mjcf_data = self.from_mjcf(mjcf_file)
        
        self._parents = self.mjcf_data["parent_indices"]
        self.model_names = self.mjcf_data["node_names"]
        self._offsets = self.mjcf_data["local_translation"][None, ].to(device)
        self._local_rotation = self.mjcf_data["local_rotation"][None, ].to(device)
        self.n_joints = len(self._parents)
        
        
    def from_mjcf(self, path):
        # function from poselib
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("Could not find worldbody in the mjcf file.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("Could not find body in the mjcf file.")
        
        # create variables to store the data
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        
        # recursively adding the nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            # parse the node name
            node_name = xml_node.attrib.get["name"]
            # parse the local translation and rotation 
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            # append the data
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            # parse the joint range
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None:
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            # recursively add the children
            curr_index = node_index
            node_index += 1
            for child in xml_node.findall("body"):
                node_index = _add_xml_node(child, curr_index, node_index)
            # return the next index
            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": parent_indices,
            "local_translation": local_translation,
            "local_rotation": local_rotation,
            "joints_ranges": joints_range
        }
        
    def fk_batch(self, pose, trans):
        # configure info
        device, dtype = pose.device, pose.dtype
        # get the shape of the pose
        batch_size, sequance_length = pose.shape[:2] # [batch_size, sequence_length, num_joints, 3]
        pose = pose[..., :self.n_joints, :] # make sure the pose is the right shape
        # convert the pose to rotation matrix
        pose_quat = tRot.axis_angle_to_quaternion(pose)
        pose_mat = tRot.quaternion_to_matrix(pose_quat)
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(batch_size, sequance_length, self.n_joints, 3, 3)
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        
        return_dict = EasyDict()
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
        return return_dict
    
    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_rotations: (B, 4) tensor of unit quaternions describing the root joint rotations
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        # configure info
        device, dtype = rotations.device, rotations.dtype
        # get the shape of the inputs
        batch_size, sequence_length = rotations.shape[:2]
        
        # create the variables to store the global positions and rotations
        position_world = []
        rotation_world = []
        
        expanded_offsets = self._offsets.expand(batch_size, sequence_length, self.n_joints, 3).to(device).type(dtype)
        
        for joint in range(self.n_joints):
            # if is the root joint
            if self._parents[joint] == -1:
                position_world.append(root_positions)
                rotation_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotation_world[self._parents[joint]][:, :, 0], expanded_offsets[:, :, joint, :, None]).squeeze(-1) + position_world[self._parents[joint]])
                rot_mat = torch.matmul(rotation_world[self._parents[joint]], torch.matmul(self._local_rotation_mat[:,  (joint):(joint + 1)], rotations[:, :, (joint - 1):joint, :]))

                position_world.append(jpos)
                rotation_world.append(rot_mat)
                
        position_world = torch.stack(position_world, dim=2)
        rotation_world = torch.cat(rotation_world, dim=2)
        return position_world, rotation_world
        
        