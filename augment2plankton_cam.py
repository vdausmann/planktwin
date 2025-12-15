import cv2
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import os

def adjust_resolution_magnification(image, target_resolution, scale_factor):
    """
    Adjust image resolution and magnification
    
    Args:
        image: Input image (numpy array)
        target_resolution: Tuple of (width, height) for desired resolution
        scale_factor: Float representing the magnification change
    """
    if scale_factor <= 0:
        raise ValueError(f"Scale factor must be positive, got {scale_factor}")
        
    scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, 
                       interpolation=cv2.INTER_LINEAR)
    
    # Skip second resize if target_resolution is None
    if target_resolution is None:
        return scaled
        
    return cv2.resize(scaled, target_resolution, interpolation=cv2.INTER_LINEAR)

def convert_illumination(image, source_type='brightfield', target_type='darkfield'):
    """
    Convert between brightfield and darkfield illumination using image inversion
    
    Args:
        image: Input image (numpy array)
        source_type: Current illumination ('brightfield' or 'darkfield')
        target_type: Target illumination ('brightfield' or 'darkfield')
    """
    if source_type == target_type:
        return image
        
    return cv2.bitwise_not(image)

def modify_depth_of_field(image, blur_amount, kernel_size=None):
    """
    Simulate different depth of field by applying controlled blur
    
    Args:
        image: Input image (numpy array)
        blur_amount: Sigma value for Gaussian blur
        kernel_size: Optional tuple for blur kernel size
    """
    if kernel_size is None:
        # Calculate kernel size based on blur amount
        kernel_size = int(2 * round(blur_amount) + 1)
        kernel_size = (kernel_size, kernel_size)
    
    return cv2.GaussianBlur(image, kernel_size, blur_amount)

# Add these configurations after existing functions
SOURCE_CONFIG = {
    'pixel_size': 27.6,  # micrometers per pixel
    'illumination': 'darkfield',
    'color_mode': 'color'
}

# Target configurations
TARGET_CONFIGS = {
    'brightfield_gray': {
        'pixel_sizes': [23, 50, 75, 100],
        'illumination': 'brightfield',
        'color_mode': 'gray'
    },
    'darkfield_color': {
        'pixel_sizes': [23, 50, 75, 100],
        'illumination': 'darkfield',
        'color_mode': 'color'
    }
}

def calculate_scale_factor(source_pixel_size: float, target_pixel_size: float) -> float:
    """Calculate the scale factor between source and target pixel sizes"""
    return source_pixel_size / target_pixel_size

def process_image(image_path: str, output_dir: str, target_config: Dict) -> None:
    """
    Process a single image for all configurations in target_config
    
    Args:
        image_path: Path to source image
        output_dir: Directory to save transformed images
        target_config: Dictionary containing target configuration
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale if needed
    if target_config['color_mode'] == 'gray' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    base_name = Path(image_path).stem
    
    # Process for each pixel size
    for pixel_size in target_config['pixel_sizes']:
        # Calculate scale factor
        scale_factor = calculate_scale_factor(SOURCE_CONFIG['pixel_size'], pixel_size)
        
        # Prepare parameters for transformation
        target_params = {
            'resolution': None,  # Will maintain aspect ratio
            'magnification_factor': scale_factor,
            'source_illumination': SOURCE_CONFIG['illumination'],
            'target_illumination': target_config['illumination'],
            'blur_sigma': 0  # Not using blur as requested
        }
        
        # Transform the image
        transformed = transform_image(img, target_params)
        
        # Apply EDOF simulation for 23um brightfield configuration
        if (pixel_size == 23 and 
            target_config['illumination'] == 'brightfield'):
            # EDOF simulation parameters
            edof_params = {
                'depth': 50,
                'dov': 1,
                'blur_multiplier': 2,
                'noise_strength': 30,
                'edge_strength': 0.3,
                'translucency_threshold': 0.2,  # 10% brightness threshold
                'opacity_darkness': 0.8  # How dark opaque objects should be (0-1)
            }

            transformed = edof_sim(
                transformed,
                depth=edof_params['depth'],
                dov=edof_params['dov'],
                blur_multiplier=edof_params['blur_multiplier'],
                noise_strength=edof_params['noise_strength'],
                edge_strength=edof_params['edge_strength'],
                translucency_threshold=edof_params['translucency_threshold'],
                opacity_darkness=edof_params['opacity_darkness']
            )
            
            # Update output filename to indicate EDOF
            output_filename = f"{base_name}_{target_config['illumination']}_{target_config['color_mode']}_{pixel_size}um_edof.png"
        else:
            output_filename = f"{base_name}_{target_config['illumination']}_{target_config['color_mode']}_{pixel_size}um.png"
        
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, transformed)

def process_directory(input_dir: str, output_dir: str, image_name: str = None) -> None:
    """
    Process a specific image from the input directory
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save transformed images
        image_name: Name of the specific image to process
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the specific image
    image_path = Path(input_dir) / image_name
    if not image_path.exists():
        raise ValueError(f"Image not found: {image_path}")
    
    print(f"Processing image: {image_path.name}")
    # Process for each target configuration
    for config_name, config in TARGET_CONFIGS.items():
        process_image(str(image_path), output_dir, config)

# Example usage:
def transform_image(input_image, target_params):
    """
    Apply all transformations in sequence
    
    Args:
        input_image: Source image
        target_params: Dictionary with target camera parameters
    """
    img = input_image.copy()
    
    # 1. Adjust resolution and magnification
    img = adjust_resolution_magnification(
        img, 
        target_params['resolution'],
        target_params['magnification_factor']
    )
    
    # 2. Convert illumination type
    img = convert_illumination(
        img,
        target_params['source_illumination'],
        target_params['target_illumination']
    )
    
    # 3. Modify depth of field
    img = modify_depth_of_field(
        img,
        target_params['blur_sigma']
    )
    
    return img

def edof_sim(
    img: np.ndarray,
    depth: int,
    dov: int,
    blur_multiplier: int,
    noise_strength: int,
    edge_strength: float = 0.0,
    translucency_threshold: float = 0.1,  # 10% brightness threshold
    opacity_darkness: float = 0.9  # How dark opaque objects should be (0-1)
) -> np.ndarray:
    """Takes an image and returns an edof simulation with softer edges and dark opaque regions.
    
    Args:
        img (np.ndarray): gray scale image for simulation
        depth (int): depth of simulated volume
        dov (int): depth of view (focal plane span)
        blur_multiplier (int): blur strength multiplier
        noise_strength (int): noise strength (0-255)
        edge_strength (float): strength of edge preservation (0-1)
        translucency_threshold (float): threshold for opaque vs translucent (0-1)
        opacity_darkness (float): darkness factor for opaque regions (0-1)

    Returns:
        np.ndarray: Simulated EDOF image with preserved edges and dark opaque regions
    """
    # Random z-position for the object
    random_z_pos = np.random.uniform(0, depth)
    print(f"Random z-position: {random_z_pos}")
    
    # Normalize image to 0-1 range for threshold comparison
    img_norm = img.astype(np.float32) / 255.0
    
    # Estimate background intensity (assuming corners are background)
    corners = [img_norm[0:10, 0:10].mean(), 
              img_norm[0:10, -10:].mean(),
              img_norm[-10:, 0:10].mean(), 
              img_norm[-10:, -10:].mean()]
    bg_intensity = np.mean(corners)
    
    # Create opacity mask based on threshold
    opacity = np.zeros_like(img_norm)
    bright_regions = img_norm > (bg_intensity * (1 + translucency_threshold))
    dark_regions = img_norm < (bg_intensity * (1 - translucency_threshold))
    opacity[bright_regions | dark_regions] = 1.0
    
    # Smooth the opacity mask more aggressively
    opacity = cv2.GaussianBlur(opacity, (9, 9), 2.0)
    
    # Create darkened version for opaque regions with smoother transition
    darkened = img_norm * (1 - opacity * opacity_darkness)
    darkened = (darkened * 255).astype(np.uint8)
    
    # Only detect and use edges if edge_strength > 0
    if edge_strength > 0:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = edges / edges.max()
        # Smooth edges
        edges = cv2.GaussianBlur(edges, (5, 5), 1.0)
    else:
        edges = np.zeros_like(img_norm)
    
    res = np.zeros(img.shape, dtype=np.float64)
    counter = 0
    
    # Move focal plane through volume
    for t in np.arange(0, 2 * np.pi, 0.1):
        counter += 1
        f_z = depth / 2 * (1 - np.cos(t))
        
        z_dist_object = abs(f_z - random_z_pos)
        noise = np.random.sample(img.shape) * noise_strength
        
        if z_dist_object > dov:
            blurred = cv2.blur(
                darkened,
                (
                    blur_multiplier * round(z_dist_object),
                    blur_multiplier * round(z_dist_object),
                ),
            )
            
            # Use opacity for blending, with reduced influence of edges
            blend_weights = opacity + (edges * edge_strength * 0.3)
            blend_weights = np.clip(blend_weights, 0, 1)
            
            blend = (1 - blend_weights) * blurred + \
                   blend_weights * darkened.astype(np.float64)
            res += blurred
        else:
            res += darkened.astype(np.float64)
            
        res += noise
        res[np.where(res > 255 * counter)] = 255 * counter

    res /= counter
    return res.astype(np.uint8)

# Example usage
if __name__ == "__main__":
    input_directory = "/Users/vdausmann/Planktwin_camera/Nerf_sub/Shrimp/1_undistort_rotate_sub/"
    output_directory = "transformed_images"
    image_name = "1352.jpg"  # Replace with your image name
    process_directory(input_directory, output_directory, image_name)