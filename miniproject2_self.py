import matplotlib.pyplot as plt
import numpy as np
import cv2

def image_processing(image):
    img=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img_resize=cv2.resize(img,(500,500))
    img_blur=cv2.GaussianBlur(img_resize,(3,3),0)
    img_edge=cv2.Canny(img_blur, 40, 180)
    y_indices, x_indices = np.where(img_edge>0)
    return x_indices, y_indices

def image_coordinates(x_indices, y_indices):
    x_max=x_indices.max()
    y_max=y_indices.max()
    x_min=x_indices.min()
    y_min=y_indices.min()
    x_img=((x_indices-x_min)/(x_max-x_min)) * 4 + 1
    y_img=((y_indices-y_min)/(y_max-y_min))*(-4) + 5
    return x_img, y_img

def sorting(x_img, y_img):
    img_coord = np.stack((x_img,y_img),axis=1)
    coord_init=img_coord[0]
    img_coord_sorted=[]
    img_coord_sorted.append(coord_init)
    img_coord=np.delete(img_coord,0,axis=0)
    while len(img_coord)>0:
        distances = np.sum((img_coord - coord_init) ** 2, axis=1)
        min_arg=np.argmin(distances)
        coord_init=img_coord[min_arg]
        img_coord_sorted.append(coord_init)
        img_coord=np.delete(img_coord,min_arg,axis=0)
    img_coord_sorted = np.array(img_coord_sorted)
    x_coord_sorted = img_coord_sorted[:,0]
    y_coord_sorted = img_coord_sorted[:,1]
    return x_coord_sorted,y_coord_sorted
        
def forward_kinematics(theta_1, theta_2, L1, L2):
    x_joint=L1*np.cos(theta_1)
    y_joint=L1*np.sin(theta_1)
    x_ee=x_joint + L2*np.cos(theta_1+theta_2)
    y_ee=y_joint + L2*np.sin(theta_1+theta_2)
    return x_joint, y_joint, x_ee, y_ee

def inverse_kinematics(x_coord, y_coord, L1, L2):
    cos_theta_2 = np.clip(((x_coord*x_coord + y_coord*y_coord) - (L1*L1 + L2*L2))/(2*L1*L2),-1.0,1.0)
    theta_2 = np.arccos(cos_theta_2)
    theta_1_y_component = (L1 + L2 * np.cos(theta_2)) * y_coord - (L2 * np.sin(theta_2)) * x_coord
    theta_1_x_component = (L1 + L2 * np.cos(theta_2)) * x_coord + (L2 * np.sin(theta_2)) * y_coord
    
    theta_1 = np.arctan2(theta_1_y_component, theta_1_x_component)
    return theta_1, theta_2    

x_indices, y_indices = image_processing('iit_kgp_logo.png')
x_img, y_img=image_coordinates(x_indices, y_indices)
x_coord_sorted, y_coord_sorted = sorting(x_img,y_img)
L1=4
L2=5
Kp=80
theta_1_current, theta_2_current=inverse_kinematics(x_coord_sorted[0],y_coord_sorted[0],L1,L2)
plt.ion()
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-1, L1+L2)
ax.set_ylim(-1, L1+L2)
ax.set_aspect('equal')
for i in range(0, len(x_coord_sorted), 6):
    ax.clear()
    theta_1, theta_2 = inverse_kinematics(x_coord_sorted[i],y_coord_sorted[i],L1,L2)
    error_1 = theta_1 - theta_1_current
    error_2 = theta_2 - theta_2_current
    error_1 = (error_1 + np.pi) % (2 * np.pi) - np.pi
    error_2 = (error_2 + np.pi) % (2 * np.pi) - np.pi
    theta_1_current += Kp * error_1 * 0.01
    theta_2_current += Kp * error_2 * 0.01
    x_joint, y_joint, x_ee, y_ee = forward_kinematics(theta_1_current, theta_2_current, L1, L2)
    ax.plot(x_coord_sorted, y_coord_sorted, 'r--', linewidth=1)
    ax.plot(x_coord_sorted[:i+1], y_coord_sorted[:i+1], 'b-', linewidth=2)
    ax.plot([0, x_joint, x_ee],[0, y_joint, y_ee], 'g-', marker='o', markersize=5, linewidth=4)
    ax.scatter(x_ee,y_ee,s=60,color='red')
    plt.pause(0.01)
plt.ioff()
plt.show()

