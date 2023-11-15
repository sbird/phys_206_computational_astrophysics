import numpy as np
from scipy.spatial import KDTree

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
theta = 0.5      # Controls the accuracy of the simulation

# Bodies properties
positions = np.random.rand(1000, 3)  # Random positions for 1000 bodies in 3D space
masses = np.random.rand(1000)        # Random masses
velocities = np.zeros_like(positions) # Initial velocities

# Compute the center of mass and total mass for a set of bodies
def compute_com(positions, masses):
    total_mass = np.sum(masses)
    center_of_mass = np.sum(positions.T * masses, axis=1) / total_mass
    return center_of_mass, total_mass

# Approximate the force on a body using the Barnes-Hut algorithm
def barnes_hut_force(body_index, positions, masses, tree):
    # Initialize force vector
    force = np.zeros(3)

    # List of indices to check, starting with the root node
    check_indices = [tree.root]
    while check_indices:
        node_index = check_indices.pop()
        node = tree.data[node_index]

        # Compute distance to the node's center of mass
        com, total_mass = compute_com(tree.data[node.indices], masses[node.indices])
        d = positions[body_index] - com
        distance = np.linalg.norm(d)

        # Compute s/d ratio
        if tree.node_size[node_index] / distance < theta:
            # If s/d is small enough, approximate the bodies in this node as a single body
            # Avoid self-interaction
            if body_index not in node.indices:
                # Calculate force using the center of mass and total mass of the node
                force += G * total_mass * d / (distance**3)
        else:
            # If s/d is too large, check the children of this node
            if node.left is not None:
                check_indices.append(node.left)
            if node.right is not None:
                check_indices.append(node.right)

    return force

# Build the KDTree
tree = KDTree(positions)

# Compute forces on all bodies
for i in range(len(positions)):
    force = barnes_hut_force(i, positions, masses, tree)
    # Update velocities and positions here using the computed force
    # Remember to integrate the motion equations (e.g., using Euler or Verlet integration)
