import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# ==========================================
# STEP 1: IMAGE PROCESSING (OpenCV)
# ==========================================
def extract_logo_points(image_path):
    """
    Loads the logo image, finds its contours/edges, scales them, 
    and importantly, SORTS them so the robot traces continuously.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fallback placeholder
        t = np.linspace(0, 2*np.pi, 400)
        return 2.0 + 1.5 * np.cos(t), 4.5 + 1.5 * np.sin(t)

    img = cv2.resize(img, (400, 400)) # Slower sorting needs slightly fewer points
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    
    y_indices, x_indices = np.where(edges > 0)
    
    # Scale to robot space
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    x_scaled = ((x_indices - x_min) / (x_max - x_min) * 2.0 - 1.0) * 1.5 + 2.0
    y_scaled = (1.0 - (y_indices - y_min) / (y_max - y_min) * 2.0) * 1.5 + 4.5
    
    # Combine into an Nx2 array of points
    points = np.column_stack((x_scaled, y_scaled))
    
    # --- NEAREST NEIGHBOR SORTING ENGINE ---
    print("Ordering points for smooth tracing... please wait a moment.")
    sorted_points = []
    current_point = points[0]
    sorted_points.append(current_point)
    points = np.delete(points, 0, axis=0)
    
    while len(points) > 0:
        # Find the distance from the current point to all remaining points
        distances = np.sum((points - current_point) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        
        # Move to the closest point
        current_point = points[closest_idx]
        sorted_points.append(current_point)
        points = np.delete(points, closest_idx, axis=0)
        
    sorted_points = np.array(sorted_points)
    print("Sorting complete!")
    
    return sorted_points[:, 0], sorted_points[:, 1]

# Change 'iit_kgp_logo.png' to your actual filename
x_targets, y_targets = extract_logo_points('iit_kgp_logo.png')
num_steps = len(x_targets)

# ==========================================
# STEP 2: ROBOT KINEMATICS (Math Core)
# ==========================================
L1 = 5.0  # Length of inner arm link
L2 = 4.0  # Length of outer arm link

def forward_kinematics(theta1, theta2):
    """Calculates where the joints are physically located given the angles."""
    x_elbow = L1 * np.cos(theta1)
    y_elbow = L1 * np.sin(theta1)
    x_ee = x_elbow + L2 * np.cos(theta1 + theta2)
    y_ee = y_elbow + L2 * np.sin(theta1 + theta2)
    return x_ee, y_ee, x_elbow, y_elbow

def inverse_kinematics(x, y):
    """Calculates what the angles need to be to reach a target (x, y)."""
    r_sq = x**2 + y**2
    r = np.sqrt(r_sq)
    
    # Safety Check: If point is too far away or too close, the arm can't reach it
    if r > (L1 + L2) or r < abs(L1 - L2):
        return None, None
        
    cos_theta2 = np.clip((r_sq - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    theta2 = np.arccos(cos_theta2) # Elbow-down solution
    
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return theta1, theta2

# ==========================================
# STEP 3: INITIALIZE CONTROL ENVIRONMENT
# ==========================================
th1_init, th2_init = inverse_kinematics(x_targets[0], y_targets[0])
current_theta = np.array([th1_init, th2_init])

Kp = 150.0       # Proportional gain (how aggressively the arm chases errors)
dt = 0.005       # Simulated time step (seconds per frame)

drawn_x, drawn_y = [], []

# ==========================================
# STEP 4: VISUALIZATION (Matplotlib Window)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 8)
ax.set_ylim(-1, 9)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("2-DOF Kinematic Simulation Tracing Logo Outlines")

# Visual line components
line_target, = ax.plot(x_targets, y_targets, color='gray', alpha=0.3, linestyle='', marker='.', label="Target Points")
line_trail, = ax.plot([], [], 'crimson', linewidth=2, label="Traced Line")
line_arm, = ax.plot([], [], 'o-', color='#1f77b4', linewidth=4, markersize=8, label="Robot Arm")

ax.legend(loc="upper left")

# ==========================================
# STEP 5: THE LIVE EXECUTION LOOP
# ==========================================
def update(frame):
    global current_theta
    
    x_d = x_targets[frame]
    y_d = y_targets[frame]
    
    th1_d, th2_d = inverse_kinematics(x_d, y_d)
    if th1_d is None: return line_arm, line_trail
    desired_theta = np.array([th1_d, th2_d])
    
    # Sub-stepping / Micro-interpolation loop for high accuracy
    # This runs the math 5 times per visual frame to catch minute details
    for _ in range(5):
        error = desired_theta - current_theta
        error = (error + np.pi) % (2 * np.pi) - np.pi 
        
        theta_dot = Kp * error                 
        current_theta += theta_dot * (dt / 5)  # Split the time step into micro-steps
    
    # Find actual physical positions to draw
    x_ee, y_ee, x_elbow, y_elbow = forward_kinematics(current_theta[0], current_theta[1])
    
    # Record the trail history
    drawn_x.append(x_ee)
    drawn_y.append(y_ee)
    
    # Refresh the visual plot lines
    line_arm.set_data([0, x_elbow, x_ee], [0, y_elbow, y_ee])
    line_trail.set_data(drawn_x, drawn_y)
    
    return line_arm, line_trail

# Launch the simulation loop
ani = FuncAnimation(fig, update, frames=num_steps, interval=15, blit=True, repeat=False)
plt.show()