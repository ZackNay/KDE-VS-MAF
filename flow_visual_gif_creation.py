import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Set seed
np.random.seed(42)

#Generate initial Gaussian distribution
n_points = 2000
initial_dist = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[0.5, 0], [0, 0.5]],
    size=n_points
)

#Generate target distribution
theta = np.linspace(0, 4*np.pi, n_points//2)
radius = np.linspace(0.2, 2, n_points//2)
noise = 0.1

#Create spiral arms
def create_spiral(theta, radius, offset=0, noise=0.1):
    x = radius * np.cos(theta + offset)
    y = radius * np.sin(theta + offset)
    #Add noise
    x += np.random.normal(0, noise * (1 + radius/2), len(radius))
    y += np.random.normal(0, noise * (1 + radius/2), len(radius))
    return np.column_stack([x, y])

#Create main spiral and secondary spiral
spiral1 = create_spiral(theta, radius)
spiral2 = create_spiral(theta, radius * 0.8, offset=np.pi)

#Combine into target distribution
target_dist = np.vstack([spiral1, spiral2])

def coupling_layer(x, t):
    """Coupling layer transformation"""
    out = x.copy()
    radius = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    angle = np.arctan2(x[:, 1], x[:, 0])
    
    # Spiral effect
    scale = 1 + t * np.sin(angle + radius)
    shift = t * radius * np.cos(angle)
    
    out[:, 1] = x[:, 1] * scale + shift
    return out

def elementwise_flow(x, t):
    """Element-wise transformation"""
    radius = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    angle = np.arctan2(x[:, 1], x[:, 0])
    
    #Smooth radial expansion
    new_radius = radius * (1 + 0.5 * t * np.sin(2 * angle))
    
    return np.column_stack([
        new_radius * np.cos(angle + t * radius),
        new_radius * np.sin(angle + t * radius)
    ])

def affine_flow(x, t):
    """Affine transformation with smooth rotation and scaling"""
    #Rotation matrix
    theta = t * np.pi / 2
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    #Smooth scaling
    radius = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    scale = 1 + 0.3 * t * (1 + np.sin(radius))
    
    return (x @ rotation.T) * scale[:, np.newaxis]

#Create figure
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter([], [], alpha=0.6, s=5, c='blue')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

def sigmoid(x, k=10):
    """Smooth transition function"""
    return 1 / (1 + np.exp(-k * (x - 0.5)))

def get_title(t):
    if t < 0.33:
        return 'Coupling Layer Transform'
    elif t < 0.66:
        return 'Element-wise Flow'
    else:
        return 'Affine Flow'

def init():
    scatter.set_offsets(initial_dist)
    ax.set_title('Initial Gaussian Distribution', pad=20, fontsize=14)
    return [scatter]

def update(frame):
    t = frame / 100
    points = initial_dist.copy()
    
    #Coupling layer
    t1 = sigmoid(t * 3) if t < 0.33 else 1
    points = coupling_layer(points, t1)
    
    #Element-wise flow
    t2 = sigmoid((t - 0.33) * 3) if 0.33 <= t < 0.66 else (0 if t < 0.33 else 1)
    points = elementwise_flow(points, t2)
    
    #Affine flow and blend with target
    t3 = sigmoid((t - 0.66) * 3) if t >= 0.66 else 0
    points = affine_flow(points, t3)
    
    if t >= 0.66:
        #Smooth interpolation to target distribution
        blend = sigmoid((t - 0.66) * 5)
        points = (1 - blend) * points + blend * target_dist
    
    #Update scatter plot
    scatter.set_offsets(points)
    colors = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    scatter.set_array(colors)

    ax.set_title(get_title(t), pad=20, fontsize=14)
    
    return [scatter]

#Create animation
anim = FuncAnimation(fig, update, frames=100, init_func=init,
                    interval=50, blit=True)
plt.tight_layout()

#Save animation
anim.save('normalizing_flows_galaxy.gif', writer='pillow')
plt.show()