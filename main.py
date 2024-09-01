import streamlit as st
import numpy as np
import re
from typing import Dict, Tuple, List

def hex_to_rgb(hex_color):
    """Convert hex color code to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    # Convert the hex color code to a tuple of integers (R, G, B)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def parse_css_to_dict(file_path):
    """Parse a CSS file and convert CSS variables to a dictionary with RGB values."""
    # Read the CSS file content
    with open(file_path, 'r') as file:
        css_content = file.read()

    # Dictionary to store the variables
    css_dict = {}

    # Regular expression to find CSS variables in the :root block
    pattern = r'--([\w-]+):\s*#([a-fA-F0-9]{6});'

    # Find all matches and store them in the dictionary
    matches = re.findall(pattern, css_content)
    for match in matches:
        css_dict[match[0]] = hex_to_rgb(f"#{match[1]}")

    return css_dict

# Specify the path to your .css file
css_file_path = 'pantone-colors-variables.css'

# Parse the CSS content and convert it to a dictionary with RGB values
color_dict = parse_css_to_dict(css_file_path)



def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color code to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Calculate the Euclidean distance between two RGB colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

def find_closest_colors(input_color: Tuple[int, int, int], color_dict: Dict[str, Tuple[int, int, int]], k: int) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Find the k closest colors from the color dictionary to the input color."""
    distances = {name: color_distance(input_color, rgb) for name, rgb in color_dict.items()}
    closest_colors = sorted(distances.items(), key=lambda item: item[1])[:k]
    return [(name, color_dict[name]) for name, _ in closest_colors]

# Streamlit app
st.title('Color Matcher')

# User input for color
color_input = st.text_input('Enter a color (hex or RGB):', '#ffffff')
k = st.slider('Number of closest colors to show:', min_value=1, max_value=10, value=3)

# Process input color
try:
    if color_input.startswith('#'):
        input_rgb = hex_to_rgb(color_input)
    else:
        input_rgb = tuple(map(int, re.findall(r'\d+', color_input)))
    
    st.write(f'Input color RGB: {input_rgb}')
    st.color_picker(label="Your Input Color", value=f'#{input_rgb[0]:02x}{input_rgb[1]:02x}{input_rgb[2]:02x}', disabled=True)
    
    # Find closest colors
    closest_colors = find_closest_colors(input_rgb, color_dict, k)
    
    # Display results
    st.write('Closest colors:')
    for name, rgb in closest_colors:
        st.write(f'{name}: RGB {rgb}')
        st.color_picker(label=name, value=f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}', disabled=True)

except Exception as e:
    st.error(f'Error processing input: {e}')

