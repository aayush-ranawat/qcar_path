import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import csv
import math

# ============================================================
# Interactive SDCS Trajectory Editor
# ============================================================
#
# FEATURES
# ------------------------------------------------------------
# - Generate Quanser SDCS roadmap path
# - Downsample dense path into uniformly spaced control points
# - Drag points with mouse
# - Add waypoint: Shift + Left Click
# - Delete waypoint: Right Click
# - Smooth spline visualization
# - Save dense smoothed trajectory
#
# OUTPUTS
# ------------------------------------------------------------
# edited_path.npy          -> shape (2, N)
# edited_waypoints.csv
#
# CONTROLS
# ------------------------------------------------------------
# Left Click + Drag   -> Move point
# Shift + Left Click  -> Add point
# Right Click         -> Delete point
# Press 's'           -> Save path
#
# ============================================================

from hal.products.mats import SDCSRoadMap

# ============================================================
# USER SETTINGS
# ============================================================

useSmallMap = False
leftHandTraffic = False

nodeSequence = [0, 2, 4, 14 , 20 , 22 , 10]

# Choose the exact number of uniformly spaced control points
NUM_CONTROL_POINTS = 30

# Final dense trajectory size
FINAL_PATH_POINTS = 4000

# ============================================================
# PATH SOURCE SETTINGS
# ============================================================

# True  -> load existing .npy path
# False -> generate SDCS roadmap path
LOAD_PATH_FROM_NPY = True

# Existing saved path
NPY_PATH_FILE = "edited_path.npy"

def yaw_from_quaternion(x, y, z, w):
    """
    Calculate the yaw (rotation around Z-axis) from a quaternion.
    Returns the angle in radians between -pi and +pi.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return yaw


# ============================================================
# Generate or Load Path
# ============================================================

if LOAD_PATH_FROM_NPY:

    print("\nLoading path from:", NPY_PATH_FILE)

    dense_path = np.load(NPY_PATH_FILE)

    print("Loaded path shape:", dense_path.shape)

else:

    roadmap = SDCSRoadMap(
        leftHandTraffic=leftHandTraffic,
        useSmallMap=useSmallMap
    )

    initialPose = roadmap.get_node_pose(
        nodeSequence[0]
    ).squeeze()

    # Shape -> (2, N)
    dense_path = roadmap.generate_path(
        nodeSequence=nodeSequence
    )[:2, :]

    print("\nGenerated SDCS roadmap path")
    print("Dense path shape:", dense_path.shape)


# ============================================================
# Downsample for Editing (Arclength Interpolation)
# ============================================================

# 1. Calculate the distance between consecutive points
differences = np.diff(dense_path, axis=1)
distances = np.linalg.norm(differences, axis=0)

# 2. Calculate the cumulative distance along the path
cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
total_length = cumulative_distances[-1]

# 3. Create target distances that are perfectly uniform
uniform_distances = np.linspace(0, total_length, NUM_CONTROL_POINTS)

# 4. Interpolate the X and Y coordinates at those specific distances
uniform_x = np.interp(uniform_distances, cumulative_distances, dense_path[0, :])
uniform_y = np.interp(uniform_distances, cumulative_distances, dense_path[1, :])

# Shape -> (NUM_CONTROL_POINTS, 2)
points = np.vstack((uniform_x, uniform_y)).T.copy()

print("\n================================================")
print("Total path length    : {:.2f} m".format(total_length))
print("Distance between pts : {:.2f} m".format(total_length / (NUM_CONTROL_POINTS - 1)))
print("Editable control pts :", len(points))
print("================================================")

selected_idx = None


# ============================================================
# Create Figure
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_title("Interactive SDCS Waypoint Editor")

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

ax.grid(True)
ax.axis("equal")


# ============================================================
# Generate Smooth Path
# ============================================================

def generate_smooth_path():

    if len(points) < 4:
        return points.T

    try:

        tck, u = splprep(
            [points[:, 0], points[:, 1]],
            s=0
        )

        unew = np.linspace(0, 1, FINAL_PATH_POINTS)

        smooth = splev(unew, tck)

        return np.vstack([smooth[0], smooth[1]])

    except Exception as e:

        print("Spline generation failed:", e)

        return points.T


# ============================================================
# Draw Everything
# ============================================================

def draw():

    ax.clear()

    ax.set_title(
        "Drag=Move | Shift+Click=Add | RightClick=Delete | S=Save"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    ax.grid(True)
    ax.axis("equal")

    # ========================================================
    # Smooth Path
    # ========================================================

    smooth_path = generate_smooth_path()

    # Dense smooth trajectory
    ax.plot(
        smooth_path[0],
        smooth_path[1],
        linewidth=2,
        label='Smoothed Path'
    )

    # Editable control points
    ax.plot(
        points[:, 0],
        points[:, 1],
        'ro-',
        markersize=6,
        linewidth=1,
        label='Control Points'
    )

    # Draw waypoint numbers
    for i, p in enumerate(points):

        ax.text(
            p[0],
            p[1],
            str(i),
            fontsize=9
        )

    ax.legend()

    fig.canvas.draw_idle()


# ============================================================
# Find Closest Point
# ============================================================

def get_nearest_point(event, threshold=0.15):

    if event.xdata is None or event.ydata is None:
        return None

    mouse = np.array([event.xdata, event.ydata])

    distances = np.linalg.norm(points - mouse, axis=1)

    idx = np.argmin(distances)

    if distances[idx] < threshold:
        return idx

    return None


# ============================================================
# Mouse Press
# ============================================================

def on_press(event):

    global selected_idx
    global points

    if event.inaxes != ax:
        return

    # ========================================================
    # LEFT CLICK
    # ========================================================

    if event.button == 1:

    # ----------------------------------------------------
    # Display clicked coordinates
    # ----------------------------------------------------

        draw.clicked_point = (
            event.xdata,
            event.ydata
        )

        print(
            f"Clicked Point: "
            f"({event.xdata:.3f}, {event.ydata:.3f})"
        )

    # ----------------------------------------------------
    # SHIFT + CLICK -> Add point
    # ----------------------------------------------------

        if event.key == 'shift':

            new_point = np.array([
                [event.xdata, event.ydata]
            ])

            insert_idx = len(points)

            min_dist = float('inf')

            for i in range(len(points) - 1):

                midpoint = 0.5 * (points[i] + points[i + 1])

                dist = np.linalg.norm(
                    midpoint - new_point[0]
                )

                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1

            points = np.insert(
                points,
                insert_idx,
                new_point,
                axis=0
            )

            draw()

            return

        # ----------------------------------------------------
        # Select existing point
        # ----------------------------------------------------

        idx = get_nearest_point(event)

        if idx is not None:
            selected_idx = idx

    # ========================================================
    # RIGHT CLICK -> Delete point
    # ========================================================

    elif event.button == 3:

        idx = get_nearest_point(event)

        if idx is not None and len(points) > 4:

            points = np.delete(
                points,
                idx,
                axis=0
            )

            draw()


# ============================================================
# Mouse Release
# ============================================================

def on_release(event):

    global selected_idx

    selected_idx = None


# ============================================================
# Mouse Move
# ============================================================

def on_motion(event):

    global selected_idx
    global points

    if selected_idx is None:
        return

    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    points[selected_idx] = [
        event.xdata,
        event.ydata
    ]

    draw()


# ============================================================
# Save Path
# ============================================================

def save_path():

    smooth_path = generate_smooth_path()

    # ========================================================
    # Save NumPy
    # ========================================================

    np.save(
        "edited_path.npy",
        smooth_path
    )

    # ========================================================
    # Save CSV
    # ========================================================

    with open(
        "edited_waypoints.csv",
        "w",
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["x", "y"])

        for i in range(smooth_path.shape[1]):

            writer.writerow([
                smooth_path[0, i],
                smooth_path[1, i]
            ])

    print("\n================================================")
    print("Saved:")
    print("  edited_path.npy")
    print("  edited_waypoints.csv")
    print("Shape:", smooth_path.shape)
    print("================================================")


# ============================================================
# Keyboard Events
# ============================================================

def on_key(event):

    if event.key == 's':
        save_path()


# ============================================================
# Connect Events
# ============================================================

fig.canvas.mpl_connect(
    'button_press_event',
    on_press
)

fig.canvas.mpl_connect(
    'button_release_event',
    on_release
)

fig.canvas.mpl_connect(
    'motion_notify_event',
    on_motion
)

fig.canvas.mpl_connect(
    'key_press_event',
    on_key
)

# ============================================================
# Initial Draw
# ============================================================

draw()

plt.show()