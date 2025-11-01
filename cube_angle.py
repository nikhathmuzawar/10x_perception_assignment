#!/usr/bin/env python3
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import struct

class CuboidRotationAnalyzer:
    def __init__(self, bag_path):
        #Initialize the analyzer with the ROS bag path.

        self.bag_path = bag_path
        self.depth_images = []
        self.timestamps = []
        self.results = []
    
    def read_rosbag(self):
       #Read depth images from ROS bag file.
        print(f"Reading ROS bag from: {self.bag_path}")
        bag_path = Path(self.bag_path)
        if bag_path.suffix == '.db3':
            bag_dir = bag_path.parent if bag_path.parent.name else Path('.')
            db3_file = bag_path
        print(f"Bag directory: {bag_dir}")
        
        try:
            with Reader(bag_dir) as reader:
                # Create typestore for deserialization
                typestore = get_typestore(Stores.ROS2_HUMBLE)
                
                # Print available topics for debugging
                print("\nAvailable topics:")
                for conn in reader.connections:
                    print(f"  - {conn.topic} ({conn.msgtype})")
                
                # Get all connections for /depth topic
                connections = [c for c in reader.connections if c.topic == '/depth']
                
                if not connections:
                    raise ValueError("No /depth topic found in bag file")
                
                print(f"\nReading messages from /depth topic...")
                
                # Read messages
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    try:
                        # Deserialize the message using typestore
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    except Exception as e:
                        print(f"Warning: Could not deserialize message: {e}")
                        continue
                    
                    # Extract depth image data
                    depth_image = self.extract_depth_image(msg)
                    
                    if depth_image is not None:
                        self.depth_images.append(depth_image)
                        self.timestamps.append(timestamp * 1e-9)  # Convert to seconds
                        
        except Exception as e:
            print(f"Error reading bag file: {e}")
            raise
                    
        print(f"Loaded {len(self.depth_images)} depth images")
        
    def extract_depth_image(self, msg):
        #Extract depth image from sensor_msgs/Image message.
        
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        
        # Handle different depth encodings
        if encoding == '32FC1':  # 32-bit float, single channel
            depth_data = np.frombuffer(msg.data, dtype=np.float32)
        elif encoding == '16UC1':  # 16-bit unsigned int (millimeters)
            depth_data = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
        else:
            print(f"Warning: Unsupported encoding {encoding}, attempting 32FC1")
            depth_data = np.frombuffer(msg.data, dtype=np.float32)
        
        # Reshape to image dimensions
        depth_image = depth_data.reshape((height, width))
        
        # Replace invalid values (inf, nan, 0) with a large number
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        
        return depth_image
    
    def segment_cuboid(self, depth_image):
        #Finds the region of the cuboid in the depth image using depth thresholding and morphology.
        
        # Remove zero/invalid depth values
        valid_depth = depth_image[depth_image > 0.1]  # Minimum valid depth
        
        if len(valid_depth) == 0:
            return None
        
        # Use depth statistics to find the object
        mean_depth = np.mean(valid_depth)
        std_depth = np.std(valid_depth)
        
        # Threshold: object is closer than mean - std
        threshold = mean_depth - 0.5 * std_depth
        mask = (depth_image > 0.1) & (depth_image < threshold)
        mask = mask.astype(np.uint8) * 255
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find the largest contour (assuming it's the cuboid)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        return final_mask
    
    def detect_multiple_planes(self, points_3d, threshold=0.01, min_points=100):
        #Detect multiple planes in the 3D point cloud using RANSAC iteratively
        
        planes = []
        remaining_points = points_3d.copy()
        remaining_indices = np.arange(len(points_3d))
        
        max_planes = 3 
        
        for plane_idx in range(max_planes):
            if len(remaining_points) < min_points:
                break
            
            # RANSAC for plane detection
            best_normal = None
            best_inliers_mask = None
            best_inliers_count = 0
            best_point = None
            num_iterations = 150
            
            for _ in range(num_iterations):
                if len(remaining_points) < 3:
                    break
                
                # Randomly sample 3 points
                idx = np.random.choice(len(remaining_points), 3, replace=False)
                sample_points = remaining_points[idx]
                
                # Compute plane normal using cross product
                v1 = sample_points[1] - sample_points[0]
                v2 = sample_points[2] - sample_points[0]
                normal = np.cross(v1, v2)
                norm_length = np.linalg.norm(normal)
                
                if norm_length < 1e-8:
                    continue
                
                normal = normal / norm_length
                
                # Ensure normal points towards camera (negative Z)
                if normal[2] > 0:
                    normal = -normal
                
                # Count inliers
                point_on_plane = sample_points[0]
                distances = np.abs(np.dot(remaining_points - point_on_plane, normal))
                inliers_mask = distances < threshold
                inliers_count = np.sum(inliers_mask)
                
                if inliers_count > best_inliers_count:
                    best_inliers_count = inliers_count
                    best_normal = normal
                    best_inliers_mask = inliers_mask
                    best_point = point_on_plane
            
            # Check if we found a valid plane
            if best_normal is None or best_inliers_count < min_points:
                break
            
            # Store plane information with global indices
            global_inliers_mask = np.zeros(len(points_3d), dtype=bool)
            global_inliers_mask[remaining_indices[best_inliers_mask]] = True
            
            planes.append({
                'normal': best_normal,
                'inliers_mask': global_inliers_mask,
                'point': best_point,
                'num_inliers': best_inliers_count
            })
            
            # Remove inliers from remaining points
            remaining_points = remaining_points[~best_inliers_mask]
            remaining_indices = remaining_indices[~best_inliers_mask]
        
        return planes
    
    def calculate_plane_area(self, points_3d, plane_mask, fx, fy):
        #Calculate the visible area of a plane.
        plane_points = points_3d[plane_mask]
        
        if len(plane_points) == 0:
            return 0.0
        
        #Pixel counting with perspective correction
        pixel_count = np.sum(plane_mask)
        avg_depth = np.mean(plane_points[:, 2])
        
        pixel_width = avg_depth / fx
        pixel_height = avg_depth / fy
        pixel_area = pixel_width * pixel_height
        
        area_method1 = pixel_count * pixel_area
        
        # Project to plane and calculate 2D area
        # This is more accurate for non-frontal planes
        # We'll use the convex hull of projected points
        try:
            from scipy.spatial import ConvexHull
            
            # Project points onto the plane coordinate system
            hull = ConvexHull(plane_points[:, :2])
            area_method2 = hull.volume  # In 2D, volume is area
            
            # Use average of both methods
            return (area_method1 + area_method2) / 2.0
        except:
            return area_method1
    
    def estimate_plane_normal(self, depth_image, mask, visualize=True):
        #Estimate the normal vector of the largest visible face using multi-plane detection.
        height, width = depth_image.shape

        fx = fy = width * 0.8  # Approximate focal length
        cx, cy = width / 2, height / 2

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        mask_bool = mask > 0
        u_masked = u[mask_bool]
        v_masked = v[mask_bool]
        z_masked = depth_image[mask_bool]

        x = (u_masked - cx) * z_masked / fx
        y = (v_masked - cy) * z_masked / fy
        z = z_masked

        points_3d = np.stack([x, y, z], axis=1)

        if len(points_3d) < 3:
            return None, 0.0

        # Detect multiple planes
        planes = self.detect_multiple_planes(points_3d, threshold=0.015, min_points=50)
        if not planes:
            return None, 0.0

        largest_plane = None
        largest_area = 0.0
        plane_colors = [
            (0, 0, 255),   # red
            (0, 255, 0),   # green
            (255, 0, 0),   # blue
        ]

        # --- Visualization setup ---
        if visualize:
            depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.cvtColor(depth_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis_mask = np.zeros_like(depth_vis)

        for idx, plane in enumerate(planes):
            area = self.calculate_plane_area(points_3d, plane['inliers_mask'], fx, fy)
            plane['area'] = area

            if area > largest_area:
                largest_area = area
                largest_plane = plane

            # --- Visualization overlay ---
            if visualize:
                # Map plane points back to pixel coordinates
                color = plane_colors[idx % len(plane_colors)]
                plane_points = points_3d[plane['inliers_mask']]
                # Project 3D points to pixel coordinates
                u_proj = (plane_points[:, 0] * fx / plane_points[:, 2] + cx).astype(int)
                v_proj = (plane_points[:, 1] * fy / plane_points[:, 2] + cy).astype(int)
                valid = (u_proj >= 0) & (u_proj < width) & (v_proj >= 0) & (v_proj < height)
                vis_mask[v_proj[valid], u_proj[valid]] = color

        if visualize:
            blended = cv2.addWeighted(depth_vis, 0.6, vis_mask, 0.7, 0)
            cv2.imshow("Detected Planes", blended)
            cv2.waitKey(0)  # display non-blocking

        if largest_plane is None:
            return None, 0.0

        return largest_plane['normal'], largest_area

    
    def calculate_normal_angle(self, normal_vector):
        
        #Calculate the angle between the surface normal and the camera normal (Z-axis).

        camera_normal = np.array([0, 0, -1])  # Camera looks in -Z direction
        
        # Calculate angle using dot product
        cos_angle = np.dot(normal_vector, camera_normal)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def process_all_frames(self):
        #Loops through every frame to estimate normal angles and visible areas.
        
        for idx, depth_image in enumerate(self.depth_images):
            print(f"Processing frame {idx + 1}/{len(self.depth_images)}")
            
            # Segment cuboid
            mask = self.segment_cuboid(depth_image)
            
            if mask is None:
                print(f"  Warning: Could not segment cuboid in frame {idx}")
                self.results.append({
                    'frame': idx,
                    'timestamp': self.timestamps[idx],
                    'normal_angle': None,
                    'visible_area': None
                })
                continue
            
            # Estimate plane normal and visible area
            normal_vector, visible_area = self.estimate_plane_normal(depth_image, mask)
            
            if normal_vector is None:
                print(f"  Warning: Could not estimate normal in frame {idx}")
                self.results.append({
                    'frame': idx,
                    'timestamp': self.timestamps[idx],
                    'normal_angle': None,
                    'visible_area': None
                })
                continue
            
            # Calculate normal angle
            normal_angle = self.calculate_normal_angle(normal_vector)
            
            self.results.append({
                'frame': idx,
                'timestamp': self.timestamps[idx],
                'normal_angle': normal_angle,
                'visible_area': visible_area,
                'normal_vector': normal_vector
            })
            cv2.destroyAllWindows()
            print(f"  Normal angle: {normal_angle:.2f}°, Visible area: {visible_area:.4f} m²")
    
    def estimate_rotation_axis(self):
        #Estimate the axis of rotation by analyzing the change in normal vectors over time.

        print("\nEstimating rotation axis...")
        
        # Collect all valid normal vectors
        normals = []
        for result in self.results:
            if result['normal_angle'] is not None:
                normals.append(result['normal_vector'])
        
        if len(normals) < 2:
            print("Not enough valid frames to estimate rotation axis")
            return None
        
        normals = np.array(normals)
        
        # Method 1: The rotation axis should be perpendicular to all normals
        # Find the axis that minimizes the sum of squared dot products with normals
        
        # Use PCA on the normals - the rotation axis is perpendicular to the plane of normals
        centroid = np.mean(normals, axis=0)
        centered = normals - centroid
        _, _, vh = np.linalg.svd(centered)
        
        # The rotation axis is the singular vector with smallest variance
        rotation_axis = vh[2]
        
        # Normalize
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Method 2: Cross products between consecutive normals
        # The rotation axis should be aligned with these cross products
        cross_products = []
        for i in range(len(normals) - 1):
            cp = np.cross(normals[i], normals[i + 1])
            if np.linalg.norm(cp) > 1e-6:
                cross_products.append(cp / np.linalg.norm(cp))
        
        if cross_products:
            # Average cross products to get rotation axis estimate
            avg_cross = np.mean(cross_products, axis=0)
            rotation_axis_method2 = avg_cross / np.linalg.norm(avg_cross)
            
            # Choose the method that gives more consistent results
            # Check alignment between both methods
            alignment = np.abs(np.dot(rotation_axis, rotation_axis_method2))
            
            if alignment < 0.8:
                print(f"  Warning: Two methods give different results (alignment: {alignment:.2f})")
            
            # Use cross product method as it's more direct
            rotation_axis = rotation_axis_method2
        
        print(f"  Estimated rotation axis: [{rotation_axis[0]:.4f}, {rotation_axis[1]:.4f}, {rotation_axis[2]:.4f}]")
        
        return rotation_axis
    
    def save_results(self, output_dir="output"):
        #Save results to files.

        import os
        import csv
        os.makedirs(output_dir, exist_ok=True)
        
        # Save table of results to CSV
        csv_file = os.path.join(output_dir, "normal_angles_and_areas.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Frame Number', 'Timestamp (s)', 'Normal Angle (degrees)', 'Visible Area (m²)'])
            
            # Write data
            for result in self.results:
                frame = result['frame']
                timestamp = result['timestamp']
                angle = result['normal_angle']
                area = result['visible_area']
                
                if angle is not None:
                    writer.writerow([frame, f"{timestamp:.6f}", f"{angle:.2f}", f"{area:.6f}"])
                else:
                    writer.writerow([frame, f"{timestamp:.6f}", 'N/A', 'N/A'])
        
        print(f"\nResults table saved to: {csv_file}")

def main():
    bag_path = "New_assesment/depth/depth.db3" 
    # Check if bag exists (handle both .db3 file and directory)
    bag_path_obj = Path(bag_path)
    
    # Try different path variations
    if bag_path_obj.exists():
        pass  # Path exists as given
    elif Path(str(bag_path) + ".db3").exists():
        bag_path = str(bag_path) + ".db3"
    elif bag_path_obj.suffix == '.db3' and bag_path_obj.parent.exists():
        bag_path = str(bag_path_obj.parent)  # Use parent directory
    else:
        print(f"Error: ROS bag not found at {bag_path}")
        print("\nUsage: python cuboid_rotation_analysis.py <path_to_bag>")
        print("\nExamples:")
        print("  python cuboid_rotation_analysis.py depth_data.bag")
        print("  python cuboid_rotation_analysis.py depth_data.db3")
        print("  python cuboid_rotation_analysis.py depth_data  (directory)")
        return
    
    print(f"Using bag path: {bag_path}\n")
    
    # Create analyzer
    analyzer = CuboidRotationAnalyzer(bag_path)
    
    try:
        # Read ROS bag
        analyzer.read_rosbag()
        
        # Process all frames
        analyzer.process_all_frames()
        
        # Save results
        analyzer.save_results()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()