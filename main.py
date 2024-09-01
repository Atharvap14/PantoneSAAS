import streamlit as st
import numpy as np
import re
from typing import Dict, Tuple, List
import numpy as np

def ciede2000(lab1, lab2, kL=1, kC=1, kH=1):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Step 1: Calculate C1 and C2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    
    # Step 2: Calculate mean C' and G
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    # Step 3: Calculate a'1 and a'2
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # Step 4: Calculate C'1 and C'2
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    # Step 5: Calculate h'1 and h'2
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    # Step 6: Calculate ΔL', ΔC', and ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    delta_h_prime = 0
    if C1_prime * C2_prime != 0:
        delta_h_prime = h2_prime - h1_prime
        if delta_h_prime > 180:
            delta_h_prime -= 360
        elif delta_h_prime < -180:
            delta_h_prime += 360
    
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime) / 2.0)
    
    # Step 7: Calculate L', C', and H'
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    h_bar_prime = (h1_prime + h2_prime) / 2.0
    if np.abs(h1_prime - h2_prime) > 180:
        h_bar_prime += 180
    
    # Step 8: Calculate T
    T = 1 - 0.17 * np.cos(np.radians(h_bar_prime - 30)) + \
        0.24 * np.cos(np.radians(2 * h_bar_prime)) + \
        0.32 * np.cos(np.radians(3 * h_bar_prime + 6)) - \
        0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    
    # Step 9: Calculate Δθ and R_C
    delta_theta = 30 * np.exp(-((h_bar_prime - 275) / 25)**2)
    R_C = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    
    # Step 10: Calculate S_L, S_C, S_H
    S_L = 1 + (0.015 * (L_bar_prime - 50)**2) / np.sqrt(20 + (L_bar_prime - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    
    # Step 11: Calculate R_T
    R_T = -np.sin(2 * np.radians(delta_theta)) * R_C
    
    # Step 12: Calculate ΔE00
    delta_E00 = np.sqrt(
        (delta_L_prime / (kL * S_L))**2 +
        (delta_C_prime / (kC * S_C))**2 +
        (delta_H_prime / (kH * S_H))**2 +
        R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )
    
    return delta_E00
def rgb_to_xyz(rgb):
    # Normalize the RGB values to the range [0, 1]
    rgb = np.array(rgb) / 255.0
    
    # Apply the gamma correction
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92
    
    # Convert to XYZ using the standard sRGB transformation matrix
    rgb = rgb * 100  # Scale by 100 for the XYZ conversion
    transformation_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(transformation_matrix, rgb)
    return xyz

def xyz_to_lab(xyz):
    # Reference white D65
    white_ref = np.array([95.047, 100.000, 108.883])
    
    # Normalize XYZ by the reference white
    xyz = xyz / white_ref
    
    # Apply the non-linear transformation
    mask = xyz > 0.008856
    xyz[mask] = np.cbrt(xyz[mask])
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)
    
    # Calculate L, a, b
    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])
    
    return [L, a, b]

def rgb_to_lab(rgb):
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab

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
    return ciede2000(rgb_to_lab(rgb1),rgb_to_lab(rgb2))

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

