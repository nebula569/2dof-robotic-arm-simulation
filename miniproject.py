import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# IMAGE PROCESSING
def extract_logo_points(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        t = np.linspace(0, 2*np.pi, 400)
        return 2.0 + 1.5 * np.cos(t), 4.5 + 1.5 * np.sin(t)

    img = cv2.resize(img, (400, 400))
    edges = cv2.Canny(img, threshold1=50, threshold2=150)

    y_indices, x_indices = np.where(edges > 0)

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    x_scaled = ((x_indices - x_min) / (x_max - x_min) * 2.0 - 1.0) * 1.5 + 2.0
    y_scaled = (1.0 - (y_indices - y_min) / (y_max - y_min) * 2.0) * 1.5 + 4.5

    points = np.column_stack((x_scaled, y_scaled))

    print("Ordering points for smooth tracing... please wait a moment.")
    sorted_points = []
    current_point = points[0]
    sorted_points.append(current_point)
    points = np.delete(points, 0, axis=0)

    while len(points) > 0:
        distances = np.sum((points - current_point) ** 2, axis=1)
        closest_idx = np.argmin(distances)

        current_point = points[closest_idx]
        sorted_points.append(current_point)
        points = np.delete(points, closest_idx, axis=0)

    sorted_points = np.array(sorted_points)
    print("Sorting complete!")

    return sorted_points[:, 0], sorted_points[:, 1]

x_targets, y_targets = extract_logo_points('iit_kgp_logo.png')
num_steps = len(x_targets)

# ROBOT KINEMATICS
L1 = 5.0
L2 = 4.0

def forward_kinematics(theta1, theta2):
    x_elbow = L1 * np.cos(theta1)
    y_elbow = L1 * np.sin(theta1)
    x_ee = x_elbow + L2 * np.cos(theta1 + theta2)
    y_ee = y_elbow + L2 * np.sin(theta1 + theta2)
    return x_ee, y_ee, x_elbow, y_elbow

def inverse_kinematics(x, y):
    r_sq = x**2 + y**2
    r = np.sqrt(r_sq)

    if r > (L1 + L2) or r < abs(L1 - L2):
        return None, None

    cos_theta2 = np.clip((r_sq - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)

    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return theta1, theta2

# INITIALIZING CONTROL ENVIRONMENT
th1_init, th2_init = inverse_kinematics(x_targets[0], y_targets[0])
current_theta = np.array([th1_init, th2_init])

Kp = 150.0
dt = 0.005

drawn_x, drawn_y = [], []

# VISUALIZATION (Matplotlib Window)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 8)
ax.set_ylim(-1, 9)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("2-DOF Kinematic Simulation Tracing Logo Outlines")

line_target, = ax.plot(x_targets, y_targets, color='gray', alpha=0.3, linestyle='', marker='.', label="Target Points")
line_trail, = ax.plot([], [], 'crimson', linewidth=2, label="Traced Line")
line_arm, = ax.plot([], [], 'o-', color='#1f77b4', linewidth=4, markersize=8, label="Robot Arm")

ax.legend(loc="upper left")

# THE LIVE EXECUTION LOOP
def update(frame):
    global current_theta

    x_d = x_targets[frame]
    y_d = y_targets[frame]

    th1_d, th2_d = inverse_kinematics(x_d, y_d)
    if th1_d is None: return line_arm, line_trail
    desired_theta = np.array([th1_d, th2_d])

    for _ in range(5):
        error = desired_theta - current_theta
        error = (error + np.pi) % (2 * np.pi) - np.pi

        theta_dot = Kp * error
        current_theta += theta_dot * (dt / 5)

    x_ee, y_ee, x_elbow, y_elbow = forward_kinematics(current_theta[0], current_theta[1])

    drawn_x.append(x_ee)
    drawn_y.append(y_ee)

    line_arm.set_data([0, x_elbow, x_ee], [0, y_elbow, y_ee])
    line_trail.set_data(drawn_x, drawn_y)

    return line_arm, line_trail

ani = FuncAnimation(fig, update, frames=num_steps, interval=15, blit=True, repeat=False)
plt.show()
