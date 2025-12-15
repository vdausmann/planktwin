import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from mpl_toolkits.mplot3d import Axes3D

# Calibration constant
PIXEL_SIZE_UM = 27.3  # micrometers per pixel
PIXEL_SIZE_MM = PIXEL_SIZE_UM / 1000  # mm per pixel

class PlanktonImages:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.image_paths = {
            1: sorted(glob.glob(str(self.base_path / "1_undistort_rotate_sub/*.jpg"))),
            2: sorted(glob.glob(str(self.base_path / "2_undistort_rotate_sub/*.jpg"))),
            3: sorted(glob.glob(str(self.base_path / "3_undistort_rotate_sub/*.jpg"))),
            4: sorted(glob.glob(str(self.base_path / "4_undistort_rotate_sub/*.jpg")))
        }
        
    def get_synchronized_images(self, frame_idx):
        """Get four synchronized images from all cameras."""
        images = {}
        for cam_id in range(1, 5):
            if frame_idx < len(self.image_paths[cam_id]):
                img_path = self.image_paths[cam_id][frame_idx]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images[cam_id] = img
        return images

def extract_contours(images, threshold=30, area_thresh=50):
    """Extract and sort contours from images."""
    contours_all = {}
    centroids_all = {}
    
    for cam_id, img in images.items():
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, binary = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > area_thresh]
        
        # Calculate centroids
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        
        # Sort by y-coordinate
        if centroids:
            sorted_indices = np.argsort([c[1] for c in centroids])
            contours_all[cam_id] = [contours[i] for i in sorted_indices]
            centroids_all[cam_id] = [centroids[i] for i in sorted_indices]
        else:
            contours_all[cam_id] = []
            centroids_all[cam_id] = []
    
    return contours_all, centroids_all

def calculate_volume_for_pair(contours, object_idx, cam_a, cam_b, verbose=False):
    """Calculate 3D voxel volume using visual hull carving from two specific cameras.
    
    Args:
        contours: Dict of contours for all cameras
        object_idx: Index of the object to process
        cam_a: First camera ID
        cam_b: Second camera ID
        verbose: Whether to print detailed output
    
    Returns:
        Dict with volume data or None if not enough contours
    """
    if object_idx >= len(contours.get(cam_a, [])) or object_idx >= len(contours.get(cam_b, [])):
        return None
    
    ca = contours[cam_a][object_idx]
    cb = contours[cam_b][object_idx]
    
    # Get bounding boxes
    points_a = ca.reshape(-1, 2)
    points_b = cb.reshape(-1, 2)
    
    xa_min, xa_max = points_a[:, 0].min(), points_a[:, 0].max()
    ya_min, ya_max = points_a[:, 1].min(), points_a[:, 1].max()
    
    xb_min, xb_max = points_b[:, 0].min(), points_b[:, 0].max()
    yb_min, yb_max = points_b[:, 1].min(), points_b[:, 1].max()
    
    # Common Y extent
    y_min = min(ya_min, yb_min)
    y_max = max(ya_max, yb_max)
    
    if verbose:
        print(f"\nCamera pair {cam_a}-{cam_b}:")
        print(f"  Camera {cam_a}: X=[{xa_min:.0f}, {xa_max:.0f}], Y=[{ya_min:.0f}, {ya_max:.0f}]")
        print(f"  Camera {cam_b}: X=[{xb_min:.0f}, {xb_max:.0f}], Y=[{yb_min:.0f}, {yb_max:.0f}]")
        print(f"  Common Y: [{y_min:.0f}, {y_max:.0f}]")
    
    # Create binary masks from contours
    ha = int(ya_max - ya_min + 1)
    wa = int(xa_max - xa_min + 1)
    mask_a = np.zeros((ha, wa), dtype=np.uint8)
    ca_shifted = ca.copy()
    ca_shifted[:, :, 0] -= xa_min
    ca_shifted[:, :, 1] -= ya_min
    cv2.drawContours(mask_a, [ca_shifted.astype(np.int32)], -1, 255, -1)
    
    hb = int(yb_max - yb_min + 1)
    wb = int(xb_max - xb_min + 1)
    mask_b = np.zeros((hb, wb), dtype=np.uint8)
    cb_shifted = cb.copy()
    cb_shifted[:, :, 0] -= xb_min
    cb_shifted[:, :, 1] -= yb_min
    cv2.drawContours(mask_b, [cb_shifted.astype(np.int32)], -1, 255, -1)
    
    # Visual hull carving
    voxels = {}
    y_coords = np.arange(int(y_min), int(y_max) + 1)
    
    for y in y_coords:
        ya_local = int(y - ya_min)
        yb_local = int(y - yb_min)
        
        if ya_local < 0 or ya_local >= ha or yb_local < 0 or yb_local >= hb:
            continue
        
        # Get the slice for this y from both masks
        mask_a_slice = mask_a[ya_local, :]
        mask_b_slice = mask_b[yb_local, :]
        
        # Find valid coordinates
        valid_xa = np.where(mask_a_slice > 0)[0] + xa_min
        valid_xb = np.where(mask_b_slice > 0)[0] + xb_min
        
        # Create all combinations
        for xa in valid_xa:
            for xb in valid_xb:
                voxels[(int(xa), int(y), int(xb))] = True
    
    volume_px3 = len(voxels)
    volume_mm3 = volume_px3 * (PIXEL_SIZE_MM ** 3)
    volume_um3 = volume_px3 * (PIXEL_SIZE_UM ** 3)
    
    if verbose:
        print(f"  Total voxels: {volume_px3}")
        print(f"  Volume: {volume_mm3:.3f} mm³ = {volume_um3:.0f} µm³")
    
    return {
        'voxels': voxels,
        'volume_px3': volume_px3,
        'volume_mm3': volume_mm3,
        'volume_um3': volume_um3,
        'camera_pair': (cam_a, cam_b),
        'ranges': {'xa': (xa_min, xa_max), 'y': (y_min, y_max), 'xb': (xb_min, xb_max)}
    }

def compare_camera_pairs(contours, object_idx, verbose=True):
    """Calculate and compare volumes using all camera pair combinations.
    
    Camera pairs:
    - 1-3: front-left (original)
    - 1-4: front-right
    - 2-3: back-left
    - 2-4: back-right
    """
    pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Volume Calculation - All Camera Pair Combinations")
    print(f"{'='*70}")
    
    for cam_a, cam_b in pairs:
        result = calculate_volume_for_pair(contours, object_idx, cam_a, cam_b, verbose=verbose)
        if result:
            results[f"{cam_a}-{cam_b}"] = result
    
    # Calculate statistics
    if results:
        volumes_mm3 = [r['volume_mm3'] for r in results.values()]
        volumes_um3 = [r['volume_um3'] for r in results.values()]
        
        print(f"\n{'='*70}")
        print(f"STATISTICS SUMMARY")
        print(f"{'='*70}")
        print(f"\nVolumes (mm³):")
        for pair_name, result in results.items():
            print(f"  Cameras {pair_name}: {result['volume_mm3']:.3f} mm³")
        
        print(f"\nStatistics (mm³):")
        print(f"  Mean:   {np.mean(volumes_mm3):.3f} mm³")
        print(f"  Median: {np.median(volumes_mm3):.3f} mm³")
        print(f"  Std:    {np.std(volumes_mm3):.3f} mm³")
        print(f"  Min:    {np.min(volumes_mm3):.3f} mm³")
        print(f"  Max:    {np.max(volumes_mm3):.3f} mm³")
        print(f"  Range:  {np.max(volumes_mm3) - np.min(volumes_mm3):.3f} mm³")
        print(f"  CV:     {np.std(volumes_mm3) / np.mean(volumes_mm3) * 100:.1f}%")
        
        print(f"\nStatistics (µm³):")
        print(f"  Mean:   {np.mean(volumes_um3):.0f} µm³")
        print(f"  Median: {np.median(volumes_um3):.0f} µm³")
        print(f"  Std:    {np.std(volumes_um3):.0f} µm³")
        print(f"  Min:    {np.min(volumes_um3):.0f} µm³")
        print(f"  Max:    {np.max(volumes_um3):.0f} µm³")
        
    return results

def save_voxels_to_ply(voxel_data, filename):
    """Save voxels as a PLY point cloud file."""
    if voxel_data is None or not voxel_data.get('voxels'):
        return
    
    voxels = voxel_data['voxels']
    
    # Convert voxels to point cloud with real-world coordinates
    points = []
    for (x, y, z) in voxels.keys():
        # Convert to mm
        x_mm = x * PIXEL_SIZE_MM
        y_mm = y * PIXEL_SIZE_MM
        z_mm = z * PIXEL_SIZE_MM
        points.append([x_mm, y_mm, z_mm])
    
    points = np.array(points)
    
    # Write PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"Saved voxels to {filename}")

def print_voxel_extents(voxel_data):
    """Print the 3D extents of the voxel volume in mm."""
    if voxel_data is None or not voxel_data.get('ranges'):
        return
    
    ranges = voxel_data['ranges']
    
    # Handle both range formats: {'x', 'y', 'z'} or {'xa', 'y', 'xb'}
    if 'x' in ranges:
        # Format from calculate_voxel_volume
        x_min, x_max = ranges['x']
        y_min, y_max = ranges['y']
        z_min, z_max = ranges['z']
    elif 'xa' in ranges:
        # Format from calculate_volume_for_pair
        xa_min, xa_max = ranges['xa']
        y_min, y_max = ranges['y']
        xb_min, xb_max = ranges['xb']
        
        # For display purposes, use xa as x and xb as z
        x_min, x_max = xa_min, xa_max
        z_min, z_max = xb_min, xb_max
    else:
        print("Unknown range format")
        return
    
    x_extent_mm = (x_max - x_min) * PIXEL_SIZE_MM
    y_extent_mm = (y_max - y_min) * PIXEL_SIZE_MM
    z_extent_mm = (z_max - z_min) * PIXEL_SIZE_MM
    
    print(f"\n3D Extents (mm):")
    print(f"  X (width):  {x_extent_mm:.2f} mm ({x_max - x_min:.0f} px)")
    print(f"  Y (height): {y_extent_mm:.2f} mm ({y_max - y_min:.0f} px)")
    print(f"  Z (depth):  {z_extent_mm:.2f} mm ({z_max - z_min:.0f} px)")

def analyze_volume_across_frames(plankton_images, camera_pairs=[(1, 3), (1, 4), (2, 3), (2, 4)], 
                                  threshold=30, area_thresh=400):
    """Analyze volume measurements across all frames in the dataset.
    
    Args:
        plankton_images: PlanktonImages instance
        camera_pairs: List of camera pair tuples to use for volume calculation
        threshold: Threshold for contour detection
        area_thresh: Minimum contour area threshold
    
    Returns:
        Dictionary with frame-wise volume data
    """
    # Determine number of frames
    num_frames = min(len(paths) for paths in plankton_images.image_paths.values())
    
    print(f"\n{'='*70}")
    print(f"Analyzing {num_frames} frames across {len(camera_pairs)} camera pairs")
    print(f"{'='*70}\n")
    
    # Store results
    frame_data = {
        'frame_idx': [],
        'num_objects': []
    }
    
    # Add columns for each camera pair
    for cam_a, cam_b in camera_pairs:
        frame_data[f'volume_mm3_{cam_a}-{cam_b}'] = []
    
    # Process each frame
    for frame_idx in range(num_frames):
        print(f"Processing frame {frame_idx}/{num_frames-1}...", end=" ")
        
        images = plankton_images.get_synchronized_images(frame_idx)
        
        if len(images) == 4:
            contours, centroids = extract_contours(images, threshold=threshold, area_thresh=area_thresh)
            num_objects = min(len(contours[cam_id]) for cam_id in contours) if contours else 0
            
            frame_data['frame_idx'].append(frame_idx)
            frame_data['num_objects'].append(num_objects)
            
            if num_objects > 0:
                # Use first object (assuming single object per frame)
                obj_idx = 0
                
                for cam_a, cam_b in camera_pairs:
                    result = calculate_volume_for_pair(contours, obj_idx, cam_a, cam_b, verbose=False)
                    if result:
                        frame_data[f'volume_mm3_{cam_a}-{cam_b}'].append(result['volume_mm3'])
                        print(f"{cam_a}-{cam_b}: {result['volume_mm3']:.3f} mm³", end="  ")
                    else:
                        frame_data[f'volume_mm3_{cam_a}-{cam_b}'].append(np.nan)
                        print(f"{cam_a}-{cam_b}: N/A", end="  ")
                print()
            else:
                # No objects detected
                for cam_a, cam_b in camera_pairs:
                    frame_data[f'volume_mm3_{cam_a}-{cam_b}'].append(np.nan)
                print("No objects detected")
        else:
            print("Incomplete image set")
            frame_data['frame_idx'].append(frame_idx)
            frame_data['num_objects'].append(0)
            for cam_a, cam_b in camera_pairs:
                frame_data[f'volume_mm3_{cam_a}-{cam_b}'].append(np.nan)
    
    return frame_data

def plot_volume_analysis(frame_data, camera_pairs=[(1, 3), (1, 4), (2, 3), (2, 4)]):
    """Plot volume measurements across frames.
    
    Args:
        frame_data: Dictionary with frame-wise volume data
        camera_pairs: List of camera pair tuples used
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    frames = frame_data['frame_idx']
    
    # Plot 1: Volume measurements for each camera pair
    ax1 = axes[0]
    for cam_a, cam_b in camera_pairs:
        key = f'volume_mm3_{cam_a}-{cam_b}'
        volumes = frame_data[key]
        ax1.plot(frames, volumes, marker='o', label=f'Cameras {cam_a}-{cam_b}', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Volume (mm³)', fontsize=12)
    ax1.set_title('Volume Measurements Across All Frames - By Camera Pair', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean volume across camera pairs per frame
    ax2 = axes[1]
    mean_volumes = []
    std_volumes = []
    
    for i in range(len(frames)):
        frame_volumes = []
        for cam_a, cam_b in camera_pairs:
            key = f'volume_mm3_{cam_a}-{cam_b}'
            vol = frame_data[key][i]
            if not np.isnan(vol):
                frame_volumes.append(vol)
        
        if frame_volumes:
            mean_volumes.append(np.mean(frame_volumes))
            std_volumes.append(np.std(frame_volumes))
        else:
            mean_volumes.append(np.nan)
            std_volumes.append(np.nan)
    
    mean_volumes = np.array(mean_volumes)
    std_volumes = np.array(std_volumes)
    
    ax2.plot(frames, mean_volumes, marker='o', color='black', linewidth=2, markersize=6, label='Mean')
    ax2.fill_between(frames, mean_volumes - std_volumes, mean_volumes + std_volumes, 
                      alpha=0.3, color='gray', label='±1 Std Dev')
    
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Volume (mm³)', fontsize=12)
    ax2.set_title('Mean Volume Across Camera Pairs (±1 SD)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient of Variation per frame
    ax3 = axes[2]
    cv_values = []
    
    for i in range(len(frames)):
        frame_volumes = []
        for cam_a, cam_b in camera_pairs:
            key = f'volume_mm3_{cam_a}-{cam_b}'
            vol = frame_data[key][i]
            if not np.isnan(vol):
                frame_volumes.append(vol)
        
        if frame_volumes and np.mean(frame_volumes) > 0:
            cv = np.std(frame_volumes) / np.mean(frame_volumes) * 100
            cv_values.append(cv)
        else:
            cv_values.append(np.nan)
    
    ax3.plot(frames, cv_values, marker='o', color='red', linewidth=2, markersize=4)
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax3.set_title('Measurement Consistency - CV Across Camera Pairs', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% CV')
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% CV')
    ax3.legend(loc='best')
    
    plt.tight_layout()
    
    # Print overall statistics
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS ACROSS ALL FRAMES")
    print(f"{'='*70}\n")
    
    for cam_a, cam_b in camera_pairs:
        key = f'volume_mm3_{cam_a}-{cam_b}'
        volumes = [v for v in frame_data[key] if not np.isnan(v)]
        
        if volumes:
            print(f"Camera Pair {cam_a}-{cam_b}:")
            print(f"  Mean:   {np.mean(volumes):.3f} mm³")
            print(f"  Median: {np.median(volumes):.3f} mm³")
            print(f"  Std:    {np.std(volumes):.3f} mm³")
            print(f"  Min:    {np.min(volumes):.3f} mm³")
            print(f"  Max:    {np.max(volumes):.3f} mm³")
            print(f"  CV:     {np.std(volumes) / np.mean(volumes) * 100:.1f}%")
            print()
    
    # Overall statistics across all measurements
    all_volumes = []
    for cam_a, cam_b in camera_pairs:
        key = f'volume_mm3_{cam_a}-{cam_b}'
        all_volumes.extend([v for v in frame_data[key] if not np.isnan(v)])
    
    if all_volumes:
        print(f"All Measurements Combined:")
        print(f"  Total measurements: {len(all_volumes)}")
        print(f"  Mean:   {np.mean(all_volumes):.3f} mm³")
        print(f"  Median: {np.median(all_volumes):.3f} mm³")
        print(f"  Std:    {np.std(all_volumes):.3f} mm³")
        print(f"  Min:    {np.min(all_volumes):.3f} mm³")
        print(f"  Max:    {np.max(all_volumes):.3f} mm³")
        print(f"  CV:     {np.std(all_volumes) / np.mean(all_volumes) * 100:.1f}%")
    
    # Frame-to-frame mean statistics
    valid_means = [m for m in mean_volumes if not np.isnan(m)]
    if valid_means:
        print(f"\nFrame-to-Frame Mean Volume:")
        print(f"  Mean:   {np.mean(valid_means):.3f} mm³")
        print(f"  Median: {np.median(valid_means):.3f} mm³")
        print(f"  Std:    {np.std(valid_means):.3f} mm³")
        print(f"  CV:     {np.std(valid_means) / np.mean(valid_means) * 100:.1f}%")
    
    return fig

def save_voxels_to_obj(voxel_data, filename):
    """Save voxels as an OBJ file with actual cube geometry."""
    if voxel_data is None or not voxel_data.get('voxels'):
        return
    
    voxels = voxel_data['voxels']
    
    with open(filename, 'w') as f:
        f.write("# Voxel mesh\n")
        
        vertex_count = 0
        for (x, y, z) in voxels.keys():
            # Convert to mm
            x_mm = x * PIXEL_SIZE_MM
            y_mm = y * PIXEL_SIZE_MM
            z_mm = z * PIXEL_SIZE_MM
            
            # Define 8 vertices of a voxel cube
            vertices = [
                [x_mm, y_mm, z_mm],
                [x_mm + PIXEL_SIZE_MM, y_mm, z_mm],
                [x_mm + PIXEL_SIZE_MM, y_mm + PIXEL_SIZE_MM, z_mm],
                [x_mm, y_mm + PIXEL_SIZE_MM, z_mm],
                [x_mm, y_mm, z_mm + PIXEL_SIZE_MM],
                [x_mm + PIXEL_SIZE_MM, y_mm, z_mm + PIXEL_SIZE_MM],
                [x_mm + PIXEL_SIZE_MM, y_mm + PIXEL_SIZE_MM, z_mm + PIXEL_SIZE_MM],
                [x_mm, y_mm + PIXEL_SIZE_MM, z_mm + PIXEL_SIZE_MM],
            ]
            
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Define 6 faces (each face is 2 triangles)
            base = vertex_count + 1
            faces = [
                [base, base+1, base+2, base+3],  # bottom
                [base+4, base+5, base+6, base+7],  # top
                [base, base+1, base+5, base+4],  # front
                [base+2, base+3, base+7, base+6],  # back
                [base, base+3, base+7, base+4],  # left
                [base+1, base+2, base+6, base+5],  # right
            ]
            
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")
            
            vertex_count += 8
    
    print(f"Saved voxel mesh to {filename}")

def export_frame_video(plankton_images, output_path="frame_video.mp4", fps=5):
    """Export a video showing all 4 camera views with frame numbers.
    
    Args:
        plankton_images: PlanktonImages instance
        output_path: Path to save the video
        fps: Frames per second for the video
    """
    import cv2
    
    num_frames = min(len(paths) for paths in plankton_images.image_paths.values())
    
    # Get dimensions from first frame
    first_images = plankton_images.get_synchronized_images(0)
    if len(first_images) < 4:
        print("Not enough cameras for video export")
        return
    
    h, w = first_images[1].shape
    
    # Create 2x2 grid output (double width and height)
    out_h, out_w = h * 2, w * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    print(f"\nExporting video with {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        images = plankton_images.get_synchronized_images(frame_idx)
        
        if len(images) == 4:
            # Create 2x2 grid
            top_row = np.hstack([images[1], images[2]])
            bottom_row = np.hstack([images[3], images[4]])
            combined = np.vstack([top_row, bottom_row])
            
            # Convert to color for text overlay
            combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            
            # Add frame number
            text = f"Frame {frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position at top center with background
            x = (out_w - text_w) // 2
            y = 50
            cv2.rectangle(combined_color, (x-10, y-text_h-10), (x+text_w+10, y+10), (0, 0, 0), -1)
            cv2.putText(combined_color, text, (x, y), font, font_scale, (255, 255, 255), thickness)
            
            # Add camera labels
            label_font_scale = 1
            label_thickness = 2
            cv2.putText(combined_color, "Cam 1", (10, 30), font, label_font_scale, (255, 255, 255), label_thickness)
            cv2.putText(combined_color, "Cam 2", (w+10, 30), font, label_font_scale, (255, 255, 255), label_thickness)
            cv2.putText(combined_color, "Cam 3", (10, h+30), font, label_font_scale, (255, 255, 255), label_thickness)
            cv2.putText(combined_color, "Cam 4", (w+10, h+30), font, label_font_scale, (255, 255, 255), label_thickness)
            
            out.write(combined_color)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
    
    out.release()
    print(f"Video saved to {output_path}")


def analyze_single_frame(plankton_images, frame_idx, object_idx=0, camera_pair=(1, 3),
                         save_mesh=True, threshold=30, area_thresh=400, mesh_format='ply'):
    """Detailed analysis of a single frame with visualization and export.
    
    Args:
        plankton_images: PlanktonImages instance
        frame_idx: Frame index to analyze
        object_idx: Object index to analyze (default: 0, first object)
        camera_pair: Tuple of (cam_a, cam_b) for volume calculation and mesh export
        save_mesh: Whether to save voxel mesh
        threshold: Threshold for contour detection
        area_thresh: Minimum contour area threshold
        mesh_format: 'ply', 'obj', or 'both'
    """
    cam_a, cam_b = camera_pair
    
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS - Frame {frame_idx}, Object {object_idx}")
    print(f"Using Camera Pair {cam_a}-{cam_b} for 3D reconstruction")
    print(f"{'='*70}")
    print(f"Calibration: 1 pixel = {PIXEL_SIZE_UM} µm = {PIXEL_SIZE_MM} mm\n")
    
    # Load images
    images = plankton_images.get_synchronized_images(frame_idx)
    
    if len(images) != 4:
        print(f"Error: Only {len(images)} cameras available")
        return
    
    # Extract contours
    contours, centroids = extract_contours(images, threshold=threshold, area_thresh=area_thresh)
    num_objects = min(len(contours[cam_id]) for cam_id in contours) if contours else 0
    
    print(f"Detected {num_objects} objects in frame")
    
    if object_idx >= num_objects:
        print(f"Error: Object {object_idx} not found (only {num_objects} objects detected)")
        return
    
    # Calculate volumes for all camera pairs (for comparison)
    results = compare_camera_pairs(contours, object_idx, verbose=True)
    
    # Get detailed voxel data for the selected camera pair
    voxel_data = calculate_volume_for_pair(contours, object_idx, cam_a, cam_b, verbose=False)
    
    if voxel_data:
        print(f"\n{'='*70}")
        print(f"SELECTED PAIR {cam_a}-{cam_b} DETAILS")
        print(f"{'='*70}")
        print_voxel_extents(voxel_data)
        
        # Save mesh files
        if save_mesh:
            base_name = f"frame{frame_idx}_obj{object_idx}_cam{cam_a}-{cam_b}"
            
            if mesh_format in ['ply', 'both']:
                ply_name = f"{base_name}.ply"
                save_voxels_to_ply(voxel_data, ply_name)
            
            if mesh_format in ['obj', 'both']:
                obj_name = f"{base_name}.obj"
                print(f"\nWarning: OBJ export may create large files...")
                save_voxels_to_obj(voxel_data, obj_name)
    else:
        print(f"Error: Could not calculate volume for camera pair {cam_a}-{cam_b}")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Plot all 4 camera views with contours
    for i, (cam_id, img) in enumerate(sorted(images.items()), 1):
        ax = fig.add_subplot(2, 2, i)
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if cam_id in contours and object_idx < len(contours[cam_id]):
            contour = contours[cam_id][object_idx]
            
            # Highlight selected camera pair
            if cam_id in [cam_a, cam_b]:
                cv2.drawContours(img_display, [contour], -1, (0, 255, 0), 3)
            else:
                cv2.drawContours(img_display, [contour], -1, (255, 255, 0), 2)
            
            centroid = centroids[cam_id][object_idx]
            cv2.circle(img_display, centroid, 8, (255, 255, 255), -1)
            cv2.circle(img_display, centroid, 8, (0, 0, 0), 2)
            
            # Add bounding box and get dimensions
            points = contour.reshape(-1, 2)
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            cv2.rectangle(img_display, (int(x_min), int(y_min)), 
                         (int(x_max), int(y_max)), (255, 0, 0), 1)
            
            # Calculate contour properties
            area_px = cv2.contourArea(contour)
            area_mm2 = area_px * (PIXEL_SIZE_MM ** 2)
            perimeter_px = cv2.arcLength(contour, True)
            perimeter_mm = perimeter_px * PIXEL_SIZE_MM
            width_px = x_max - x_min
            height_px = y_max - y_min
            width_mm = width_px * PIXEL_SIZE_MM
            height_mm = height_px * PIXEL_SIZE_MM
            
            # Add text annotations
            info_text = (
                f"Area: {area_px:.0f} px² ({area_mm2:.3f} mm²)\n"
                f"Perimeter: {perimeter_px:.0f} px ({perimeter_mm:.2f} mm)\n"
                f"Size: {width_px:.0f} × {height_px:.0f} px\n"
                f"      ({width_mm:.2f} × {height_mm:.2f} mm)\n"
                f"Centroid: ({centroid[0]}, {centroid[1]}) px"
            )
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   family='monospace')
        
        ax.imshow(img_display)
        title = f'Camera {cam_id}'
        if cam_id in [cam_a, cam_b]:
            title += ' [SELECTED]'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_name = f'frame{frame_idx}_obj{object_idx}_cam{cam_a}-{cam_b}_analysis.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_name}")
    plt.show()
    
    return results, voxel_data

if __name__ == "__main__":
    plankton_images = PlanktonImages("/Users/vdausmann/Planktwin_camera/Nerf_sub/Seabug")
    
    # Choose mode: 'dataset' for full analysis, 'single' for single frame
    MODE = 'single'  # Change to 'single' for detailed single-frame analysis
    
    if MODE == 'dataset':
        print("="*70)
        print("MODE: Full Dataset Analysis")
        print("="*70)
        print(f"Calibration: 1 pixel = {PIXEL_SIZE_UM} µm = {PIXEL_SIZE_MM} mm")
        
        # Analyze all frames
        frame_data = analyze_volume_across_frames(plankton_images, 
                                                  camera_pairs=[(1, 3), (1, 4), (2, 3), (2, 4)],
                                                  threshold=30, 
                                                  area_thresh=400)
        
        # Plot results
        fig = plot_volume_analysis(frame_data, camera_pairs=[(1, 3), (1, 4), (2, 3), (2, 4)])
        
        # Save the plot
        fig.savefig('volume_analysis_all_frames.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved to volume_analysis_all_frames.png")
        
        # Export video of all frames
        export_frame_video(plankton_images, output_path="frames_overview.mp4", fps=5)
        
        plt.show()
        
    elif MODE == 'single':
        print("="*70)
        print("MODE: Single Frame Detailed Analysis")
        print("="*70)
        
        # Configure single frame analysis
        FRAME_IDX = 7  # Change this to analyze different frames
        OBJECT_IDX = 0  # Change this to analyze different objects
        CAMERA_PAIR = (2,4)  # Change to (1,4), (2,3), or (2,4) for different pairs
        SAVE_MESH = True
        MESH_FORMAT = 'ply'  # 'ply', 'obj', or 'both'
        
        # Run detailed analysis
        results, voxel_data = analyze_single_frame(
            plankton_images, 
            frame_idx=FRAME_IDX,
            object_idx=OBJECT_IDX,
            camera_pair=CAMERA_PAIR,
            save_mesh=SAVE_MESH,
            threshold=30,
            area_thresh=400,
            mesh_format=MESH_FORMAT
        )


##### Old unused functions for reference #####

# def get_contour_width_profile(contour, y_min, y_max):
#     """Get width (x extent) at each y coordinate for a contour."""
#     points = contour.reshape(-1, 2)
#     width_profile = {}
    
#     for y in range(int(y_min), int(y_max) + 1):
#         # Find points near this y coordinate
#         mask = np.abs(points[:, 1] - y) < 1.5
#         if np.any(mask):
#             x_coords = points[mask, 0]
#             width = x_coords.max() - x_coords.min()
#         else:
#             width = 0
#         width_profile[y] = width
    
#     return width_profile

# def calculate_voxel_volume_visual_hull(contours, object_idx):
#     """Calculate 3D voxel volume using visual hull carving from cameras 1 & 3.
    
#     1. Extract bounding boxes from both views
#     2. Normalize to common Y-extent
#     3. For each voxel, check if it projects inside both contours
#     """
#     if object_idx >= min(len(contours[cam_id]) for cam_id in [1, 3]):
#         return None
    
#     c1 = contours[1][object_idx]  # Front view (X-Y plane)
#     c3 = contours[3][object_idx]  # Side view (Z-Y plane)
    
#     # Get bounding boxes
#     points1 = c1.reshape(-1, 2)
#     points3 = c3.reshape(-1, 2)
    
#     x1_min, x1_max = points1[:, 0].min(), points1[:, 0].max()
#     y1_min, y1_max = points1[:, 1].min(), points1[:, 1].max()
    
#     z3_min, z3_max = points3[:, 0].min(), points3[:, 0].max()
#     y3_min, y3_max = points3[:, 1].min(), points3[:, 1].max()
    
#     # Common Y extent
#     y_min = min(y1_min, y3_min)
#     y_max = max(y1_max, y3_max)
    
#     print(f"\nBounding boxes:")
#     print(f"  Camera 1 (front): X=[{x1_min:.0f}, {x1_max:.0f}], Y=[{y1_min:.0f}, {y1_max:.0f}]")
#     print(f"  Camera 3 (side):  Z=[{z3_min:.0f}, {z3_max:.0f}], Y=[{y3_min:.0f}, {y3_max:.0f}]")
#     print(f"  Common Y: [{y_min:.0f}, {y_max:.0f}]")
    
#     # Create binary masks from contours
#     # For camera 1: mask in X-Y plane
#     h1 = int(y1_max - y1_min + 1)
#     w1 = int(x1_max - x1_min + 1)
#     mask1 = np.zeros((h1, w1), dtype=np.uint8)
#     c1_shifted = c1.copy()
#     c1_shifted[:, :, 0] -= x1_min
#     c1_shifted[:, :, 1] -= y1_min
#     cv2.drawContours(mask1, [c1_shifted.astype(np.int32)], -1, 255, -1)
    
#     # For camera 3: mask in Z-Y plane
#     h3 = int(y3_max - y3_min + 1)
#     w3 = int(z3_max - z3_min + 1)
#     mask3 = np.zeros((h3, w3), dtype=np.uint8)
#     c3_shifted = c3.copy()
#     c3_shifted[:, :, 0] -= z3_min
#     c3_shifted[:, :, 1] -= y3_min
#     cv2.drawContours(mask3, [c3_shifted.astype(np.int32)], -1, 255, -1)
    
#     # Visual hull carving: iterate through 3D space (VECTORIZED)
#     voxels = {}
    
#     # Create meshgrid for all possible coordinates
#     y_coords = np.arange(int(y_min), int(y_max) + 1)
#     x_coords = np.arange(int(x1_min), int(x1_max) + 1)
#     z_coords = np.arange(int(z3_min), int(z3_max) + 1)
    
#     # Check which voxels are valid
#     for y in y_coords:
#         y1_local = int(y - y1_min)
#         y3_local = int(y - y3_min)
        
#         if y1_local < 0 or y1_local >= h1 or y3_local < 0 or y3_local >= h3:
#             continue
        
#         # Get the slice for this y from both masks
#         mask1_slice = mask1[y1_local, :]
#         mask3_slice = mask3[y3_local, :]
        
#         # Find valid x and z coordinates
#         valid_x = np.where(mask1_slice > 0)[0] + x1_min
#         valid_z = np.where(mask3_slice > 0)[0] + z3_min
        
#         # Create all combinations of valid (x, z) for this y
#         for x in valid_x:
#             for z in valid_z:
#                 voxels[(int(x), int(y), int(z))] = True
    
#     volume_px3 = len(voxels)
    
#     # Convert to physical units
#     volume_mm3 = volume_px3 * (PIXEL_SIZE_MM ** 3)
#     volume_um3 = volume_px3 * (PIXEL_SIZE_UM ** 3)
    
#     print(f"\nVisual hull calculation:")
#     print(f"  Total voxels: {volume_px3}")
#     print(f"  Volume: {volume_mm3:.3f} mm³ = {volume_um3:.0f} µm³")
    
#     return {
#         'voxels': voxels,
#         'volume_px3': volume_px3,
#         'volume_mm3': volume_mm3,
#         'volume_um3': volume_um3,
#         'ranges': {'x': (x1_min, x1_max), 'y': (y_min, y_max), 'z': (z3_min, z3_max)},
#         'masks': {'mask1': mask1, 'mask3': mask3}
#     }

# def calculate_voxel_volume(contours, object_idx):
#     """Calculate 3D voxel representation and volume from 4 perpendicular views.
    
#     Camera pairs:
#     - Cameras 1 & 2: front and back (perpendicular pair)
#     - Cameras 3 & 4: left and right (perpendicular pair)
    
#     Common axis: Y (vertical)
#     """
#     if object_idx >= min(len(contours[cam_id]) for cam_id in contours):
#         return None
    
#     # Get contours for the 4 cameras
#     c1 = contours[1][object_idx]  # Front view
#     c2 = contours[2][object_idx]  # Back view (opposite of 1)
#     c3 = contours[3][object_idx]  # Left view
#     c4 = contours[4][object_idx]  # Right view (opposite of 3)
    
#     points1 = c1.reshape(-1, 2)
#     points2 = c2.reshape(-1, 2)
#     points3 = c3.reshape(-1, 2)
#     points4 = c4.reshape(-1, 2)
    
#     # Y is the common axis (vertical in all views)
#     y_min = min(points1[:, 1].min(), points2[:, 1].min(), points3[:, 1].min(), points4[:, 1].min())
#     y_max = max(points1[:, 1].max(), points2[:, 1].max(), points3[:, 1].max(), points4[:, 1].max())
    
#     # X comes from cameras 1 & 2 (their horizontal axis)
#     x_min_1 = points1[:, 0].min()
#     x_max_1 = points1[:, 0].max()
#     x_min_2 = points2[:, 0].min()
#     x_max_2 = points2[:, 0].max()
#     x_min = min(x_min_1, x_min_2)
#     x_max = max(x_max_1, x_max_2)
    
#     # Z comes from cameras 3 & 4 (their horizontal axis)
#     z_min_3 = points3[:, 0].min()
#     z_max_3 = points3[:, 0].max()
#     z_min_4 = points4[:, 0].min()
#     z_max_4 = points4[:, 0].max()
#     z_min = min(z_min_3, z_min_4)
#     z_max = max(z_max_3, z_max_4)
    
#     print(f"\nDEBUG - Axis ranges (pixels):")
#     print(f"  X: {x_min:.0f} to {x_max:.0f} (width: {x_max - x_min:.0f}px = {(x_max - x_min) * PIXEL_SIZE_MM:.2f}mm)")
#     print(f"  Y: {y_min:.0f} to {y_max:.0f} (height: {y_max - y_min:.0f}px = {(y_max - y_min) * PIXEL_SIZE_MM:.2f}mm)")
#     print(f"  Z: {z_min:.0f} to {z_max:.0f} (depth: {z_max - z_min:.0f}px = {(z_max - z_min) * PIXEL_SIZE_MM:.2f}mm)")
    
#     # Get width profiles
#     profile1 = get_contour_width_profile(c1, y_min, y_max)  # X widths at each Y
#     profile2 = get_contour_width_profile(c2, y_min, y_max)  # X widths at each Y
#     profile3 = get_contour_width_profile(c3, y_min, y_max)  # Z widths at each Y
#     profile4 = get_contour_width_profile(c4, y_min, y_max)  # Z widths at each Y
    
#     # Build 3D voxel grid
#     voxels = {}
#     volume_px3 = 0
    
#     for y in np.arange(int(y_min), int(y_max) + 1):
#         # Get X extent at this Y from cameras 1 & 2
#         width_x = (profile1.get(y, 0) + profile2.get(y, 0)) / 2
        
#         # Get Z extent at this Y from cameras 3 & 4
#         width_z = (profile3.get(y, 0) + profile4.get(y, 0)) / 2
        
#         if width_x < 1 or width_z < 1:
#             continue
        
#         # Get the center and extent for each dimension at this Y
#         x_center_1 = (points1[np.abs(points1[:, 1] - y) < 1.5, 0].min() + 
#                       points1[np.abs(points1[:, 1] - y) < 1.5, 0].max()) / 2 if np.any(np.abs(points1[:, 1] - y) < 1.5) else x_min
#         z_center_3 = (points3[np.abs(points3[:, 1] - y) < 1.5, 0].min() + 
#                       points3[np.abs(points3[:, 1] - y) < 1.5, 0].max()) / 2 if np.any(np.abs(points3[:, 1] - y) < 1.5) else z_min
        
#         # For this Y level, only fill voxels within the X and Z extents
#         x_start = int(x_center_1 - width_x / 2)
#         x_end = int(x_center_1 + width_x / 2)
#         z_start = int(z_center_3 - width_z / 2)
#         z_end = int(z_center_3 + width_z / 2)
        
#         for x in np.arange(max(x_start, x_min), min(x_end, x_max) + 1):
#             for z in np.arange(max(z_start, z_min), min(z_end, z_max) + 1):
#                 voxels[(int(x), int(y), int(z))] = True
#                 volume_px3 += 1
    
#     # Convert to physical units
#     volume_mm3 = volume_px3 * (PIXEL_SIZE_MM ** 3)
#     volume_um3 = volume_px3 * (PIXEL_SIZE_UM ** 3)
    
#     return {
#         'voxels': voxels,
#         'volume_px3': volume_px3,
#         'volume_mm3': volume_mm3,
#         'volume_um3': volume_um3,
#         'ranges': {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)},
#         'profiles': {'profile1': profile1, 'profile2': profile2, 'profile3': profile3, 'profile4': profile4}
#     }

# def visualize_voxels(voxel_data, title="3D Voxel Visualization"):
#     """Visualize the voxel grid."""
#     if voxel_data is None or not voxel_data.get('voxels'):
#         return
    
#     voxels = voxel_data['voxels']
#     ranges = voxel_data['ranges']
#     y_min, y_max = ranges['y']
#     z_min, z_max = ranges['z']
    
#     # Create boolean grid
#     grid = np.zeros((int(y_max - y_min + 1), int(z_max - z_min + 1), 100), dtype=bool)
    
#     for (x, y, z) in voxels.keys():
#         if 0 <= int(y - y_min) < grid.shape[0] and 0 <= int(z - z_min) < grid.shape[1]:
#             grid[int(y - y_min), int(z - z_min), 50] = True
    
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     ax.voxels(grid, facecolors='C0', edgecolors='k', linewidth=0.5)
#     ax.set_xlabel('Y (pixels)')
#     ax.set_ylabel('Z (pixels)')
#     ax.set_zlabel('X')
#     ax.set_title(title)
    
#     return fig, ax

# def print_object_info(contours, centroids, voxel_data, object_idx):
#     """Print information about detected object."""
#     print(f"\n{'='*60}")
#     print(f"Object {object_idx + 1}:")
#     print(f"{'='*60}")
    
#     for cam_id in sorted(contours.keys()):
#         if object_idx < len(contours[cam_id]):
#             contour = contours[cam_id][object_idx]
#             centroid = centroids[cam_id][object_idx]
            
#             points = contour.reshape(-1, 2)
#             x_min, x_max = points[:, 0].min(), points[:, 0].max()
#             y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
#             width_px = x_max - x_min
#             height_px = y_max - y_min
#             width_mm = width_px * PIXEL_SIZE_MM
#             height_mm = height_px * PIXEL_SIZE_MM
            
#             print(f"Camera {cam_id}:")
#             print(f"  Centroid: ({centroid[0]}, {centroid[1]}) px")
#             print(f"  Size: {width_px:.0f}px × {height_px:.0f}px = {width_mm:.2f}mm × {height_mm:.2f}mm")
    
#     if voxel_data:
#         print(f"\nVolume calculation:")
#         print(f"  Voxels: {voxel_data['volume_px3']:.0f} px³")
#         print(f"  Volume: {voxel_data['volume_mm3']:.3f} mm³ = {voxel_data['volume_um3']:.0f} µm³")
