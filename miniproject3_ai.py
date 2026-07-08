import matplotlib.pyplot as plt
import numpy as np
import cv2

def image_processing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fallback dummy shape (a heart/circle tracking loop) if image file is missing
        t = np.linspace(0, 2*np.pi, 300)
        return (100 + 50*np.cos(t)).astype(int), (100 + 50*np.sin(t)).astype(int)
    
    img_resize = cv2.resize(img, (150, 150)) # Kept modest for crisp real-time control physics
    img_blur = cv2.GaussianBlur(img_resize, (3,3), 0)
    img_edge = cv2.Canny(img_blur, 50, 150)
    y_indices, x_indices = np.where(img_edge > 0)
    return x_indices, y_indices

def image_coordinates(x_indices, y_indices):
    x_max, x_min = x_indices.max(), x_indices.min()
    y_max, y_min = y_indices.max(), y_indices.min()
    
    # Safely shifting and bounding the logo to sit squarely in the center of the workspace
    # Fits within X: [1.5, 4.5], Y: [1.5, 4.5]. Maximum distance from origin ~6.36 (Well below max reach 7.0)
    x_img = ((x_indices - x_min) / (x_max - x_min)) * 3.0 + 1.5
    y_img = ((y_indices - y_min) / (y_max - y_min)) * (-3.0) + 4.5
    return x_img, y_img

def sorting(x_img, y_img):
    img_coord = np.stack((x_img, y_img), axis=1)
    coord_init = img_coord[0]
    img_coord_sorted = [coord_init]
    img_coord = np.delete(img_coord, 0, axis=0)
    
    while len(img_coord) > 0:
        distances = np.sum((img_coord - coord_init) ** 2, axis=1)
        min_arg = np.argmin(distances)
        coord_init = img_coord[min_arg]
        img_coord_sorted.append(coord_init)
        img_coord = np.delete(img_coord, min_arg, axis=0)
        
    img_coord_sorted = np.array(img_coord_sorted)
    return img_coord_sorted[:, 0], img_coord_sorted[:, 1]
        
def forward_kinematics(theta_1, theta_2, L1, L2):
    x_joint = L1 * np.cos(theta_1)
    y_joint = L1 * np.sin(theta_1)
    x_ee = x_joint + L2 * np.cos(theta_1 + theta_2)
    y_ee = y_joint + L2 * np.sin(theta_1 + theta_2)
    return x_joint, y_joint, x_ee, y_ee

def inverse_kinematics(x_coord, y_coord, L1, L2):
    # Algebraic Law of Cosines configuration
    cos_theta_2 = (x_coord**2 + y_coord**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta_2 = np.clip(cos_theta_2, -1.0, 1.0)
    theta_2 = np.arccos(cos_theta_2) # Forces stable elbow-up configurations
    
    # Using 2-argument arctan to securely resolve positional quadrant tracking
    alpha = np.arctan2(y_coord, x_coord)
    beta = np.arctan2(L2 * np.sin(theta_2), L1 + L2 * np.cos(theta_2))
    theta_1 = alpha - beta
    
    return theta_1, theta_2

# --- Setup Data Pipelines ---
x_indices, y_indices = image_processing('iit_kgp_logo.png')
x_img, y_img = image_coordinates(x_indices, y_indices)
x_coord_sorted, y_coord_sorted = sorting(x_img, y_img)

L1, L2 = 3.0, 4.0

# --- Setup P-Controller Parameters ---
Kp = 15.0        # Proportional controller gain constant
dt = 0.01        # Simulated time step size per frame loop

# Initialize robot physical states at the first pixel position
theta_1_current, theta_2_current = inverse_kinematics(x_coord_sorted[0], y_coord_sorted[0], L1, L2)

# --- Matplotlib Optimization Structure (Executed ONCE) ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, L1 + L2 + 1)
ax.set_ylim(-1, L1 + L2 + 1)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("2-DOF Arm Tracking with Joint P-Controller", fontsize=10)

# Generate empty handles to modify inline later
ax.plot(x_coord_sorted, y_coord_sorted, 'k--', linewidth=1, alpha=0.2, label="Logo Target Profile")
drawn_path, = ax.plot([], [], 'b-', linewidth=2, label="Actual Trajectory")
robot_arm, = ax.plot([], [], 'g-o', linewidth=4, markersize=7, markerfacecolor='black', label="Arm Linkage")
end_effector = ax.scatter([], [], s=50, color='red', zorder=5)
ax.legend(loc="upper right", fontsize=8)

# --- Controller Execution Loop ---
# Skips every few pixels to smoothly match our dt physics loop interval rate
step_size = max(1, len(x_coord_sorted) // 250)

for i in range(0, len(x_coord_sorted), step_size):
    x_target = x_coord_sorted[i]
    y_target = y_coord_sorted[i]
    
    # 1. Compute where the joints SHOULD be mathematically
    theta_1_target, theta_2_target = inverse_kinematics(x_target, y_target, L1, L2)
    
    # 2. Compute Error Metrics
    error_1 = theta_1_target - theta_1_current
    error_2 = theta_2_target - theta_2_current
    
    # Handle angle wrap-around boundary corrections (-pi to +pi jumps)
    error_1 = (error_1 + np.pi) % (2 * np.pi) - np.pi
    error_2 = (error_2 + np.pi) % (2 * np.pi) - np.pi
    
    # 3. P-Controller Control Law output calculation
    omega_1 = Kp * error_1
    omega_2 = Kp * error_2
    
    # 4. Numerical Integration (Move the actual physics joints forward)
    theta_1_current += omega_1 * dt
    theta_2_current += omega_2 * dt
    
    # 5. Calculate real-time forward kinematics based on the P-controlled position
    x_joint, y_joint, x_ee, y_ee = forward_kinematics(theta_1_current, theta_2_current, L1, L2)
    
    # 6. Ultra-Fast Object Property Replacement (No ax.clear!)
    drawn_path.set_data(x_coord_sorted[:i+1], y_coord_sorted[:i+1])
    robot_arm.set_data([0, x_joint, x_ee], [0, y_joint, y_ee])
    end_effector.set_offsets([[x_ee, y_ee]])
    
    plt.pause(0.001)

plt.ioff()
plt.show()