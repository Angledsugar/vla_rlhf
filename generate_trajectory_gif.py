"""
결과 JSON 파일을 읽어서 액션 궤적을 시각화하는 GIF를 생성하는 스크립트.
Franka Panda 로봇의 Forward Kinematics를 사용하여 End-Effector 궤적을 계산합니다.
각 demo의 이미지와 prompt도 함께 표시합니다.
로봇팔의 실제 움직임을 3D로 시각화합니다.

Usage:
    python generate_trajectory_gif.py
    python generate_trajectory_gif.py --input results.json
    python generate_trajectory_gif.py --demo demo_0  # 특정 demo만 생성
    python generate_trajectory_gif.py --dataset /path/to/dataset  # 데이터셋 경로 지정

출력 폴더명 형식: {model}_{dataset}_{timestamp}
예: pi05_droid_hroi_20251207_143052
"""
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset" / "hribench" / "hroi"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


# ============================================================================
# Franka Panda Forward Kinematics
# ============================================================================

# Franka Panda DH Parameters (Modified DH Convention)
# Reference: https://frankaemika.github.io/docs/control_parameters.html
PANDA_DH_PARAMS = [
    # (a, d, alpha) - theta is the joint variable
    (0,      0.333,  0),         # Joint 1
    (0,      0,     -np.pi/2),   # Joint 2
    (0,      0.316,  np.pi/2),   # Joint 3
    (0.0825, 0,      np.pi/2),   # Joint 4
    (-0.0825, 0.384, -np.pi/2),  # Joint 5
    (0,      0,      np.pi/2),   # Joint 6
    (0.088,  0,      np.pi/2),   # Joint 7
]

# Flange offset (end-effector)
FLANGE_OFFSET = 0.107

# Link colors for visualization
LINK_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95190C', '#610345', '#E94F37']


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Compute the DH transformation matrix.
    
    Args:
        theta: Joint angle (radians)
        d: Link offset
        a: Link length
        alpha: Link twist
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1]
    ])


def forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    """
    Compute the end-effector position using Franka Panda Forward Kinematics.
    
    Args:
        joint_angles: Array of 7 joint angles in radians
    
    Returns:
        3D position of the end-effector [x, y, z]
    """
    # Start with identity matrix
    T = np.eye(4)
    
    # Apply each joint transformation
    for i, (a, d, alpha) in enumerate(PANDA_DH_PARAMS):
        theta = joint_angles[i]
        T = T @ dh_transform(theta, d, a, alpha)
    
    # Add flange offset (end-effector)
    T_flange = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, FLANGE_OFFSET],
        [0, 0, 0, 1]
    ])
    T = T @ T_flange
    
    # Extract position
    return T[:3, 3]


def forward_kinematics_all_joints(joint_angles: np.ndarray) -> np.ndarray:
    """
    Compute positions of all joints (including base and end-effector) for visualization.
    
    Args:
        joint_angles: Array of 7 joint angles in radians
    
    Returns:
        Array of shape [9, 3] containing positions of base, 7 joints, and end-effector
    """
    positions = np.zeros((9, 3))  # Base + 7 joints + End-effector
    
    # Base position
    positions[0] = [0, 0, 0]
    
    # Start with identity matrix
    T = np.eye(4)
    
    # Compute position after each joint
    for i, (a, d, alpha) in enumerate(PANDA_DH_PARAMS):
        theta = joint_angles[i]
        T = T @ dh_transform(theta, d, a, alpha)
        positions[i + 1] = T[:3, 3]
    
    # Add flange (end-effector)
    T_flange = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, FLANGE_OFFSET],
        [0, 0, 0, 1]
    ])
    T = T @ T_flange
    positions[8] = T[:3, 3]
    
    return positions


def compute_ee_trajectory(joint_trajectory: np.ndarray) -> np.ndarray:
    """
    Compute end-effector trajectory from joint trajectory.
    
    Args:
        joint_trajectory: Array of shape [timesteps, 7] containing joint angles
    
    Returns:
        Array of shape [timesteps, 3] containing end-effector positions
    """
    timesteps = joint_trajectory.shape[0]
    ee_trajectory = np.zeros((timesteps, 3))
    
    for t in range(timesteps):
        ee_trajectory[t] = forward_kinematics(joint_trajectory[t, :7])
    
    return ee_trajectory


def compute_all_joint_positions(joint_trajectory: np.ndarray) -> np.ndarray:
    """
    Compute all joint positions for entire trajectory.
    
    Args:
        joint_trajectory: Array of shape [timesteps, 7] containing joint angles
    
    Returns:
        Array of shape [timesteps, 9, 3] containing all joint positions
    """
    timesteps = joint_trajectory.shape[0]
    all_positions = np.zeros((timesteps, 9, 3))
    
    for t in range(timesteps):
        all_positions[t] = forward_kinematics_all_joints(joint_trajectory[t, :7])
    
    return all_positions


# ============================================================================
# Data Loading
# ============================================================================

def load_demo_image(demo_name: str, dataset_dir: Path = None) -> np.ndarray | None:
    """Load the demo image from dataset."""
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    image_path = dataset_dir / demo_name / "wrong_answers" / "processed_frame.png"
    if image_path.exists():
        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    return None


def generate_output_folder_name(input_file: str) -> str:
    """
    Generate output folder name from input filename.
    
    Expected input format: {model}_{dataset}_{timestamp}.json
    Output format: {model}_{dataset}_{timestamp}
    
    If input doesn't match expected format, uses current timestamp.
    """
    input_path = Path(input_file)
    filename = input_path.stem  # Remove .json extension
    
    # Check if filename already has expected format (contains at least 2 underscores)
    parts = filename.split('_')
    if len(parts) >= 3:
        return filename
    
    # Fallback: generate with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results_{timestamp}"


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


# ============================================================================
# Visualization
# ============================================================================

def draw_robot_arm(ax, joint_positions, gripper_state=0, alpha=1.0):
    """
    Draw the robot arm given joint positions.
    
    Args:
        ax: 3D matplotlib axis
        joint_positions: Array of shape [9, 3] with positions of all joints
        gripper_state: Gripper opening state (0=closed, 1=open)
        alpha: Transparency
    
    Returns:
        List of artists for animation
    """
    artists = []
    
    # Draw links (thick lines connecting joints)
    for i in range(8):
        line, = ax.plot(
            [joint_positions[i, 0], joint_positions[i+1, 0]],
            [joint_positions[i, 1], joint_positions[i+1, 1]],
            [joint_positions[i, 2], joint_positions[i+1, 2]],
            color=LINK_COLORS[i], linewidth=6, alpha=alpha, solid_capstyle='round'
        )
        artists.append(line)
    
    # Draw joints (spheres at joint positions)
    for i in range(9):
        if i == 0:
            # Base - larger
            point, = ax.plot([joint_positions[i, 0]], [joint_positions[i, 1]], [joint_positions[i, 2]],
                           'ko', markersize=15, alpha=alpha)
        elif i == 8:
            # End-effector - red
            point, = ax.plot([joint_positions[i, 0]], [joint_positions[i, 1]], [joint_positions[i, 2]],
                           'ro', markersize=12, alpha=alpha)
        else:
            # Regular joints
            point, = ax.plot([joint_positions[i, 0]], [joint_positions[i, 1]], [joint_positions[i, 2]],
                           'o', color='#333333', markersize=10, alpha=alpha)
        artists.append(point)
    
    # Draw simple gripper
    ee_pos = joint_positions[8]
    gripper_width = 0.02 + gripper_state * 0.03  # Width varies with gripper state
    
    # Get gripper orientation from last link direction
    link_dir = joint_positions[8] - joint_positions[7]
    link_dir = link_dir / (np.linalg.norm(link_dir) + 1e-6)
    
    # Perpendicular direction for gripper fingers
    perp = np.array([link_dir[1], -link_dir[0], 0])
    perp = perp / (np.linalg.norm(perp) + 1e-6) * gripper_width
    
    # Draw gripper fingers
    finger1, = ax.plot(
        [ee_pos[0], ee_pos[0] + perp[0]],
        [ee_pos[1], ee_pos[1] + perp[1]],
        [ee_pos[2], ee_pos[2] - 0.05],
        color='#666666', linewidth=4, alpha=alpha
    )
    finger2, = ax.plot(
        [ee_pos[0], ee_pos[0] - perp[0]],
        [ee_pos[1], ee_pos[1] - perp[1]],
        [ee_pos[2], ee_pos[2] - 0.05],
        color='#666666', linewidth=4, alpha=alpha
    )
    artists.extend([finger1, finger2])
    
    return artists


def create_trajectory_gif(
    demo_name: str,
    actions: np.ndarray,
    initial_state: dict,
    prompt: str,
    output_path: Path,
    dataset_dir: Path = None,
    frame_stride: int = 2,
    fps: int = 5,
    dpi: int = 100,
    image_only: bool = False,
    save_final_image: bool = False,
    image_format: str = "png",
):
    """
    Create a GIF visualizing the action trajectory with Franka Panda FK.
    Includes the demo image, prompt, and animated robot arm.
    Optimized for faster GIF generation.
    
    Args:
        demo_name: Name of the demo
        actions: Action array of shape [timesteps, 8] (7 joints + 1 gripper)
        initial_state: Initial joint and gripper positions
        prompt: The prompt text for this demo
        output_path: Path to save the GIF
        dataset_dir: Path to the dataset directory for loading images
        frame_stride: Use every Nth frame to cut render time (min 1)
        fps: Frames per second for the saved GIF
        dpi: Figure DPI; lower values render faster/smaller files
        image_only: If True, skip GIF and save only final frame
        save_final_image: If True, also save the final frame alongside GIF
        image_format: File extension for saved image (e.g., png, jpg)
    """
    frame_stride = max(1, int(frame_stride))
    fps = max(1, int(fps))
    dpi = max(40, int(dpi))
    image_format = (image_format or "png").lstrip(".")

    actions = np.array(actions)
    timesteps, action_dim = actions.shape
    
    # Get initial joint positions
    initial_joint = np.array(initial_state["joint_position"])
    initial_gripper = np.array(initial_state["gripper_position"])
    initial_full = np.concatenate([initial_joint, initial_gripper])
    
    # Accumulate actions to get joint trajectory
    joint_trajectory = np.zeros((timesteps + 1, action_dim))
    joint_trajectory[0] = initial_full
    for t in range(timesteps):
        joint_trajectory[t + 1] = joint_trajectory[t] + actions[t]
    
    # Compute end-effector trajectory using Forward Kinematics
    ee_trajectory = compute_ee_trajectory(joint_trajectory[:, :7])
    
    # Compute all joint positions for robot visualization
    all_joint_positions = compute_all_joint_positions(joint_trajectory[:, :7])
    
    # Load demo image (resize for faster rendering)
    demo_image = load_demo_image(demo_name, dataset_dir)
    if demo_image is not None:
        # Resize image for faster rendering
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(demo_image)
        img_pil = img_pil.resize((320, 240), PILImage.Resampling.LANCZOS)
        demo_image = np.array(img_pil)
    
    # Create figure with balanced size for readability and speed
    fig = plt.figure(figsize=(14, 9), dpi=dpi)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.12, 1.2, 0.8], hspace=0.3, wspace=0.3)
    
    # Title area with prompt
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    wrapped_prompt = wrap_text(prompt, width=100)
    ax_title.text(0.5, 0.5, f"{demo_name}\n\nPrompt: {wrapped_prompt}", 
                  transform=ax_title.transAxes, fontsize=9, 
                  verticalalignment='center', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                  wrap=True)
    
    # Subplot 1: Demo Image
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.set_title('Input Image', fontsize=10, fontweight='bold')
    if demo_image is not None:
        ax_img.imshow(demo_image)
    else:
        ax_img.text(0.5, 0.5, 'Image not found', transform=ax_img.transAxes,
                   ha='center', va='center', fontsize=10)
    ax_img.axis('off')
    
    # Subplot 2: 3D Robot Arm Animation (larger)
    ax_robot = fig.add_subplot(gs[1, 1:3], projection='3d')
    ax_robot.set_title('Franka Panda Robot Arm', fontsize=10, fontweight='bold')
    ax_robot.set_xlabel('X (m)', fontsize=8)
    ax_robot.set_ylabel('Y (m)', fontsize=8)
    ax_robot.set_zlabel('Z (m)', fontsize=8)
    
    # Subplot 3: End-Effector trajectory
    ax_ee = fig.add_subplot(gs[1, 3], projection='3d')
    ax_ee.set_title('End-Effector Trajectory', fontsize=10, fontweight='bold')
    ax_ee.set_xlabel('X (m)', fontsize=8)
    ax_ee.set_ylabel('Y (m)', fontsize=8)
    ax_ee.set_zlabel('Z (m)', fontsize=8)
    
    # Subplot 4: Joint angles over time
    ax_joints = fig.add_subplot(gs[2, 0])
    ax_joints.set_title('Joint Angles', fontsize=9, fontweight='bold')
    ax_joints.set_xlabel('Timestep', fontsize=8)
    ax_joints.set_ylabel('Angle (rad)', fontsize=8)
    
    # Subplot 5: End-Effector XYZ over time
    ax_xyz = fig.add_subplot(gs[2, 1])
    ax_xyz.set_title('End-Effector Position', fontsize=9, fontweight='bold')
    ax_xyz.set_xlabel('Timestep', fontsize=8)
    ax_xyz.set_ylabel('Position (m)', fontsize=8)
    
    # Subplot 6: Actions over time
    ax_actions = fig.add_subplot(gs[2, 2])
    ax_actions.set_title('Actions', fontsize=9, fontweight='bold')
    ax_actions.set_xlabel('Timestep', fontsize=8)
    ax_actions.set_ylabel('Action Value', fontsize=8)
    
    # Subplot 7: Gripper state
    ax_gripper = fig.add_subplot(gs[2, 3])
    ax_gripper.set_title('Gripper State', fontsize=9, fontweight='bold')
    ax_gripper.set_xlabel('Timestep', fontsize=8)
    ax_gripper.set_ylabel('Position', fontsize=8)
    
    # Colors for different joints
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    joint_labels = [f'J{i+1}' for i in range(7)]
    xyz_colors = ['red', 'green', 'blue']
    xyz_labels = ['X', 'Y', 'Z']
    
    # Plot static elements
    time_steps = np.arange(timesteps + 1)
    action_steps = np.arange(timesteps)
    
    # Plot joint angles
    for i in range(7):
        ax_joints.plot(time_steps, joint_trajectory[:, i], color=colors[i], label=joint_labels[i], alpha=0.7, linewidth=1)
    ax_joints.legend(loc='upper right', fontsize=6, ncol=2)
    ax_joints.grid(True, alpha=0.3)
    
    # Plot End-Effector XYZ
    for i in range(3):
        ax_xyz.plot(time_steps, ee_trajectory[:, i], color=xyz_colors[i], label=xyz_labels[i], linewidth=1.5)
    ax_xyz.legend(loc='upper right', fontsize=7)
    ax_xyz.grid(True, alpha=0.3)
    
    # Plot actions
    for i in range(7):
        ax_actions.plot(action_steps, actions[:, i], color=colors[i], label=joint_labels[i], alpha=0.7, linewidth=1)
    ax_actions.legend(loc='upper right', fontsize=6, ncol=2)
    ax_actions.grid(True, alpha=0.3)
    
    # Plot gripper
    ax_gripper.plot(time_steps, joint_trajectory[:, 7], color='purple', linewidth=1.5, label='Gripper')
    ax_gripper.fill_between(time_steps, 0, joint_trajectory[:, 7], alpha=0.3, color='purple')
    ax_gripper.grid(True, alpha=0.3)
    
    # Set robot arm axis limits
    all_x = all_joint_positions[:, :, 0].flatten()
    all_y = all_joint_positions[:, :, 1].flatten()
    all_z = all_joint_positions[:, :, 2].flatten()
    
    margin = 0.15
    x_range = max(all_x.max() - all_x.min(), 0.5)
    y_range = max(all_y.max() - all_y.min(), 0.5)
    z_range = max(all_z.max() - all_z.min(), 0.5)
    max_range = max(x_range, y_range, z_range)
    
    x_mid = (all_x.max() + all_x.min()) / 2
    y_mid = (all_y.max() + all_y.min()) / 2
    z_mid = (all_z.max() + all_z.min()) / 2
    
    ax_robot.set_xlim([x_mid - max_range/2 - margin, x_mid + max_range/2 + margin])
    ax_robot.set_ylim([y_mid - max_range/2 - margin, y_mid + max_range/2 + margin])
    ax_robot.set_zlim([-0.1, max_range + margin])
    
    # Set EE trajectory axis limits
    ee_margin = 0.05
    ee_x_range = ee_trajectory[:, 0].max() - ee_trajectory[:, 0].min()
    ee_y_range = ee_trajectory[:, 1].max() - ee_trajectory[:, 1].min()
    ee_z_range = ee_trajectory[:, 2].max() - ee_trajectory[:, 2].min()
    ee_max_range = max(ee_x_range, ee_y_range, ee_z_range, 0.1)
    
    ee_x_mid = (ee_trajectory[:, 0].max() + ee_trajectory[:, 0].min()) / 2
    ee_y_mid = (ee_trajectory[:, 1].max() + ee_trajectory[:, 1].min()) / 2
    ee_z_mid = (ee_trajectory[:, 2].max() + ee_trajectory[:, 2].min()) / 2
    
    ax_ee.set_xlim([ee_x_mid - ee_max_range/2 - ee_margin, ee_x_mid + ee_max_range/2 + ee_margin])
    ax_ee.set_ylim([ee_y_mid - ee_max_range/2 - ee_margin, ee_y_mid + ee_max_range/2 + ee_margin])
    ax_ee.set_zlim([ee_z_mid - ee_max_range/2 - ee_margin, ee_z_mid + ee_max_range/2 + ee_margin])
    
    # EE trajectory - draw full trajectory as static (pre-drawn)
    ax_ee.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2], 
               'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
    ax_ee.plot([ee_trajectory[0, 0]], [ee_trajectory[0, 1]], [ee_trajectory[0, 2]], 
               'go', markersize=6, label='Start')
    ax_ee.plot([ee_trajectory[-1, 0]], [ee_trajectory[-1, 1]], [ee_trajectory[-1, 2]], 
               'ro', markersize=6, label='End')
    ax_ee.legend(loc='upper left', fontsize=6)
    
    # Draw full EE trajectory on robot arm plot as static trace
    ax_robot.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2],
                  'c-', linewidth=1.5, alpha=0.6, label='EE Trajectory')
    ax_robot.plot([ee_trajectory[0, 0]], [ee_trajectory[0, 1]], [ee_trajectory[0, 2]], 
                  'go', markersize=6)
    ax_robot.plot([ee_trajectory[-1, 0]], [ee_trajectory[-1, 1]], [ee_trajectory[-1, 2]], 
                  'mo', markersize=6)
    
    # Timestep text
    timestep_text = fig.text(0.02, 0.02, '', fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Pre-create robot arm line objects for reuse (optimization)
    link_lines = []
    joint_points = []
    gripper_lines = []
    
    # Initialize link lines
    for i in range(8):
        line, = ax_robot.plot([], [], [], color=LINK_COLORS[i], linewidth=4, solid_capstyle='round')
        link_lines.append(line)
    
    # Initialize joint points
    for i in range(9):
        if i == 0:
            point, = ax_robot.plot([], [], [], 'ko', markersize=10)
        elif i == 8:
            point, = ax_robot.plot([], [], [], 'ro', markersize=8)
        else:
            point, = ax_robot.plot([], [], [], 'o', color='#333333', markersize=7)
        joint_points.append(point)
    
    # Initialize gripper lines
    finger1, = ax_robot.plot([], [], [], color='#666666', linewidth=3)
    finger2, = ax_robot.plot([], [], [], color='#666666', linewidth=3)
    gripper_lines = [finger1, finger2]
    
    def update_robot(joint_positions, gripper_state):
        """Update robot arm visualization by modifying existing line data."""
        # Update links
        for i in range(8):
            link_lines[i].set_data(
                [joint_positions[i, 0], joint_positions[i+1, 0]],
                [joint_positions[i, 1], joint_positions[i+1, 1]]
            )
            link_lines[i].set_3d_properties(
                [joint_positions[i, 2], joint_positions[i+1, 2]]
            )
        
        # Update joints
        for i in range(9):
            joint_points[i].set_data([joint_positions[i, 0]], [joint_positions[i, 1]])
            joint_points[i].set_3d_properties([joint_positions[i, 2]])
        
        # Update gripper
        ee_pos = joint_positions[8]
        gripper_width = 0.02 + abs(gripper_state) * 0.03
        
        link_dir = joint_positions[8] - joint_positions[7]
        link_norm = np.linalg.norm(link_dir)
        if link_norm > 1e-6:
            link_dir = link_dir / link_norm
        
        perp = np.array([link_dir[1], -link_dir[0], 0])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-6:
            perp = perp / perp_norm * gripper_width
        
        gripper_lines[0].set_data(
            [ee_pos[0], ee_pos[0] + perp[0]],
            [ee_pos[1], ee_pos[1] + perp[1]]
        )
        gripper_lines[0].set_3d_properties([ee_pos[2], ee_pos[2] - 0.04])
        
        gripper_lines[1].set_data(
            [ee_pos[0], ee_pos[0] - perp[0]],
            [ee_pos[1], ee_pos[1] - perp[1]]
        )
        gripper_lines[1].set_3d_properties([ee_pos[2], ee_pos[2] - 0.04])
    
    # Initialize robot at first frame
    update_robot(all_joint_positions[0], joint_trajectory[0, 7])
    timestep_text.set_text(f'Timestep: 0/{timesteps}')
    
    def animate(frame):
        # Update robot arm (reuse existing objects)
        update_robot(all_joint_positions[frame], joint_trajectory[frame, 7])
        
        # Update timestep text
        timestep_text.set_text(f'Timestep: {frame}/{timesteps}')
    
    # Downsample frames to reduce render/encode time
    frames = list(range(0, timesteps + 1, frame_stride))
    if frames[-1] != timesteps:
        frames.append(timesteps)

    # Save final frame if requested
    if image_only or save_final_image:
        final_frame = timesteps
        update_robot(all_joint_positions[final_frame], joint_trajectory[final_frame, 7])
        timestep_text.set_text(f'Timestep: {final_frame}/{timesteps}')
        image_path = output_path.with_suffix(f".{image_format}")
        fig.savefig(image_path, bbox_inches='tight')
        print(f"  Final frame saved to {image_path}")
        if not image_only:
            # Reset to first frame for animation
            first_frame = frames[0]
            update_robot(all_joint_positions[first_frame], joint_trajectory[first_frame, 7])
            timestep_text.set_text(f'Timestep: {first_frame}/{timesteps}')

    if image_only:
        plt.close(fig)
        return
    
    # Create animation with blit=False for stable 3D rendering
    anim = animation.FuncAnimation(
        fig, animate,
        frames=frames, interval=200, blit=False
    )
    
    # Save animation as GIF with optimized settings
    gif_path = output_path.with_suffix('.gif')
    try:
        pillow_writer = animation.PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=pillow_writer)
        print(f"  GIF saved to {gif_path}")
    except Exception as e:
        print(f"  Warning: Could not save GIF: {e}")
    
    plt.close(fig)


def generate_gifs(input_file: str = "hribench_results.json", demo_filter: str = None, 
                  output_dir: str = None, dataset_dir: str = None,
                  frame_stride: int = 2, fps: int = 5, dpi: int = 100,
                  image_only: bool = False, save_final_image: bool = False,
                  image_format: str = "png"):
    """
    Generate trajectory GIFs from results file.
    
    Args:
        input_file: Name of the input JSON file (or full path)
        demo_filter: If specified, only generate GIF for this demo (e.g., "demo_0")
        output_dir: Output directory for GIFs (default: auto-generated from input filename)
        dataset_dir: Path to the dataset directory for loading images/prompts
        frame_stride: Use every Nth frame to cut render/encode time
        fps: Frames per second for the saved GIF
        dpi: Figure DPI; lower values are faster/smaller
        image_only: If True, save only final frame (no GIF)
        save_final_image: If True, save final frame in addition to GIF
        image_format: File extension for saved image (e.g., png, jpg)
    """
    # Determine input path
    input_path = Path(input_file)
    if not input_path.is_absolute():
        results_path = OUTPUT_DIR / input_file
    else:
        results_path = input_path
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run run_inference.py first to generate results.")
        return
    
    # Determine dataset directory
    if dataset_dir is not None:
        data_path = Path(dataset_dir)
    else:
        data_path = DATASET_DIR
    
    # Determine output directory
    if output_dir is not None:
        video_output_dir = Path(output_dir)
    else:
        # Auto-generate folder name from input filename: {model}_{dataset}_{timestamp}
        folder_name = generate_output_folder_name(results_path.name)
        video_output_dir = OUTPUT_DIR / "videos" / folder_name
    
    # Create output directory if it doesn't exist
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Filter demos if specified
    if demo_filter:
        if demo_filter not in results:
            print(f"Error: Demo '{demo_filter}' not found in results.")
            print(f"Available demos: {list(results.keys())[:10]}...")
            return
        results = {demo_filter: results[demo_filter]}
    
    print(f"Generating GIFs for {len(results)} demos...")
    print(f"Using Franka Panda Forward Kinematics for End-Effector trajectory")
    print(f"Including animated robot arm visualization")
    print(f"Input file: {results_path}")
    print(f"Dataset directory: {data_path}")
    print(f"Output directory: {video_output_dir}")
    mode_desc = "image only" if image_only else ("gif + image" if save_final_image else "gif only")
    print(f"Frame stride: {frame_stride} | FPS: {fps} | DPI: {dpi} | Mode: {mode_desc} | Image format: {image_format}")
    print("=" * 50)
    
    for idx, (demo_name, data) in enumerate(results.items()):
        print(f"[{idx+1}/{len(results)}] Creating GIF for {demo_name}...")
        
        gif_path = video_output_dir / f"{demo_name}_trajectory.gif"
        
        # Use initial state if available, otherwise use zeros
        initial_state = data.get("initial_state", {
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0]
        })
        
        # Get prompt
        prompt = data.get("prompt", "No prompt available")
        
        create_trajectory_gif(
            demo_name,
            data["actions"],
            initial_state,
            prompt,
            gif_path,
            data_path,
            frame_stride=frame_stride,
            fps=fps,
            dpi=dpi,
            image_only=image_only,
            save_final_image=save_final_image,
            image_format=image_format,
        )
    
    print("=" * 50)
    print(f"All GIFs saved to {video_output_dir}")
    print(f"Total generated: {len(results)} GIFs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate trajectory GIFs from inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_trajectory_gif.py
    python generate_trajectory_gif.py --input pi05_droid_hroi_20251207_143052.json
    python generate_trajectory_gif.py --dataset /path/to/dataset
    python generate_trajectory_gif.py --demo demo_0

Output folder format (auto-generated from input filename):
    {model}_{dataset}_{timestamp}/
    Example: outputs/videos/pi05_droid_hroi_20251207_143052/
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="hribench_results.json",
        help="Input JSON filename or full path (default: hribench_results.json)"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Generate GIF for specific demo only (e.g., demo_0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for GIFs (default: auto-generated as outputs/videos/{model}_{dataset}_{timestamp}/)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to dataset directory containing demo_* folders for images/prompts (default: dataset/hribench/hroi)"
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Use every Nth frame when rendering (larger = faster, default: 2)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for GIF writer (default: 5)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure DPI for rendering (lower = faster/smaller, default: 100)"
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Save only the final frame image (skip GIF generation)"
    )
    parser.add_argument(
        "--save-final-image",
        action="store_true",
        help="Save the final frame image in addition to GIF"
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        help="Image format/extension for saved frame (default: png)"
    )
    
    args = parser.parse_args()
    generate_gifs(
        input_file=args.input, 
        demo_filter=args.demo, 
        output_dir=args.output_dir,
        dataset_dir=args.dataset,
        frame_stride=args.frame_stride,
        fps=args.fps,
        dpi=args.dpi,
        image_only=args.image_only,
        save_final_image=args.save_final_image,
        image_format=args.image_format,
    )
