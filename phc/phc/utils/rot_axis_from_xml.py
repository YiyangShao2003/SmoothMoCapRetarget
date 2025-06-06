import xml.etree.ElementTree as ET
import torch

def parse_rotation_axes(xml_path):
    """
    Parses the MuJoCo XML file to extract rotation axes for each joint
    defined in the actuators and returns a torch tensor.
    
    Args:
        xml_path (str): Path to the MuJoCo XML file.
        
    Returns:
        torch.Tensor: A tensor of shape (1, N, 3) where N is the number of joints.
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create a mapping from joint names to their axis vectors
    joint_axes = {}
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        axis = joint.get('axis')
        if axis:
            axis_vector = [float(x) for x in axis.split()]
            joint_axes[name] = axis_vector
    
    # Iterate over actuators and collect axes in order
    rotation_axes = []
    rotation_axes_names = []
    actuators = root.findall('.//actuator/position')
    if not actuators:
        # Handle if actuators are defined differently, e.g., type="position"
        actuators = root.findall('.//actuator/*[@type="position"]')
    
    for actuator in actuators:
        joint_name = actuator.get('joint')
        if joint_name in joint_axes:
            rotation_axes.append(joint_axes[joint_name])
            rotation_axes_names.append(joint_name)
        else:
            raise ValueError(f"Joint '{joint_name}' not found or does not have an axis defined.")
    
    # Convert to torch tensor and add batch dimension
    rotation_axes_tensor = torch.tensor([rotation_axes], dtype=torch.float32)
    
    return rotation_axes_tensor, rotation_axes_names

# Example usage
if __name__ == "__main__":
    # xml_file = "/home/yshao/2024/ws_rl/humanml3d_retargeting/resources/robots/h1/h1.xml"
    xml_file = "/home/yshao/2024/ws_rl/humanml3d_retargeting/resources/robots/bh2/robot.xml"
    try:
        ROTATION_AXIS,ROTATION_AXIS_NAMES = parse_rotation_axes(xml_file)
        print("ROTATION_AXIS:")
        print(ROTATION_AXIS)
        print("ROTATION_AXIS_NAMES:")
        print(ROTATION_AXIS_NAMES)
    except Exception as e:
        print(f"Error: {e}")