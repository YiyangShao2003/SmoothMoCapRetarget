import xml.etree.ElementTree as ET

def extract_link_names(xml_path):
    """
    Parses the MuJoCo XML file to extract all link names defined in <body> elements.

    Args:
        xml_path (str): Path to the MuJoCo XML file.

    Returns:
        list: A list of link names.
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Initialize an empty list to hold link names
        link_names = []

        # Define the namespace if present (MuJoCo might use namespaces)
        namespace = ''
        if '}' in root.tag:
            namespace = root.tag.split('}')[0] + '}'

        # Find all <body> elements within <worldbody>
        worldbody = root.find(f'{namespace}worldbody')
        if worldbody is None:
            raise ValueError("No <worldbody> element found in the XML.")

        # Recursive function to traverse and collect <body> names
        def traverse_bodies(element):
            for body in element.findall(f'{namespace}body'):
                name = body.get('name')
                if name:
                    link_names.append(name)
                # Recursively traverse child bodies
                traverse_bodies(body)

        traverse_bodies(worldbody)

        return link_names

    except ET.ParseError as pe:
        print(f"XML Parse Error: {pe}")
        return []
    except FileNotFoundError:
        print(f"File not found: {xml_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example Usage
if __name__ == "__main__":
    xml_file = "/home/yshao/2024/ws_rl/humanml3d_retargeting/resources/robots/bh2/robot.xml"
    link_names = extract_link_names(xml_file)
    print("links_names = [")
    for name in link_names:
        print(f"    '{name}',")
    print("]")