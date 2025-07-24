import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

def harmonic_potential(x):
    """Harmonic oscillator potential V(x) = 0.5 * x^2"""
    return 0.5 * x**2

def imaginary_time_propagation(x_min=-10, x_max=10, dx=0.05, dt=0.001, steps=10000): #if dt is too big then diffusion equation unstable condition, if dt is too small then 10000 steps will not be sufficient to converge the wavefunction to the ground state.
    """
    Compute ground state using Imaginary Time Propagation.
    
    Parameters:
        x_min, x_max: Spatial grid boundaries
        dx: Grid spacing
        dt: Time step size (reduced for stability)
        steps: Number of propagation steps
        
    Returns:
        E0: Ground state energy
        wavefunction: Normalized ground state wavefunction
        x_grid: Spatial grid
        V: Potential on grid
    """
    # Spatial grid
    x_grid = np.arange(x_min, x_max + dx, dx)
    N = len(x_grid)
    V = harmonic_potential(x_grid)
    
    # Constants
    hbar = 1.0
    mass = 1.0
    
    # Kinetic energy operator (3-point finite difference)
    kin_diag = (hbar**2 / (mass * dx**2)) * np.ones(N)
    kin_offdiag = (-hbar**2 / (2 * mass * dx**2)) * np.ones(N - 1)
    
    # Initial random wavefunction
    psi = np.random.rand(N)
    psi /= np.sqrt(np.sum(psi**2) * dx)  # Normalize
    # print(psi)
    
    # ITP propagation
    for _ in range(steps):
        # Apply kinetic energy operator
        # print("###########################")
        T_psi = kin_diag * psi #elementwise multiplication
        # print(T_psi)
        T_psi[:-1] += kin_offdiag * psi[1:]
        # print(T_psi)
        T_psi[1:] += kin_offdiag * psi[:-1]
        # print(T_psi)
        
        # Apply potential and compute full Hamiltonian
        H_psi = T_psi + V * psi
        
        # Euler integration step
        psi_new = psi - dt * H_psi
        
        # Normalize and update
        norm = np.sqrt(np.sum(psi_new**2) * dx)
        psi = psi_new / norm
    
    # Calculate energy using Hamiltonian application
    T_psi_final = kin_diag * psi
    T_psi_final[:-1] += kin_offdiag * psi[1:]
    T_psi_final[1:] += kin_offdiag * psi[:-1]
    H_psi_final = T_psi_final + V * psi
    E0 = np.sum(psi * H_psi_final) * dx
    
    return E0, psi, x_grid, V

def diagonalization_method(x_min=-10, x_max=10, dx=0.05, num_states=5):
    """
    Compute eigenvalues/eigenvectors using Hamiltonian diagonalization.
    
    Parameters:
        x_min, x_max: Spatial grid boundaries
        dx: Grid spacing
        num_states: Number of states to return
        
    Returns:
        eigenvalues: First num_states eigenvalues
        eigenvectors: First num_states eigenvectors (columns)
        x_grid: Spatial grid
        V: Potential on grid
    """
    # Spatial grid
    x_grid = np.arange(x_min, x_max + dx, dx)
    N = len(x_grid)
    V = harmonic_potential(x_grid)
    
    # Construct Hamiltonian (tridiagonal)
    hbar = 1.0
    mass = 1.0
    diagonal = (hbar**2) / (mass * dx**2) + V
    off_diagonal = (-hbar**2 / (2 * mass * dx**2)) * np.ones(N - 1)
    
    # Diagonalize
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal, off_diagonal, select='i', select_range=(0, num_states-1)
    )
    
    # Normalize eigenvectors
    eigenvectors = eigenvectors.T / np.sqrt(dx)
    
    return eigenvalues, eigenvectors, x_grid, V
#Usage
# Compute ground state with ITP
E0, psi0, x, V = imaginary_time_propagation()
print(f"Ground state energy (ITP): {E0:.6f}")

# Compute first 5 states with diagonalization
eigenvalues, eigenvectors, x, V = diagonalization_method(num_states=5)
print("\nEnergies (Diagonalization):")
for i, E in enumerate(eigenvalues):
    print(f"State {i}: {E:.6f} (Exact: {i + 0.5:.1f})")

# Exact energies for comparison (n + 0.5)
exact = np.arange(5) + 0.5
print("Exact energies:", exact)
plt.figure(figsize=(10, 6))
plt.plot(x, V, 'k-', lw=2, label="Potential")
for i in range(3):  # Plot first 3 states
    plt.plot(x, eigenvectors[i] + eigenvalues[i], label=f"State {i}")
plt.title("Harmonic Oscillator Eigenstates")
plt.xlabel("Position")
plt.ylabel("Energy / Wavefunction")
plt.legend()
plt.grid(True)
plt.show()
'''
Key Features:
ITP: Efficient for ground state with O(N) operations per step.
Diagonalization: Accurate for multiple states using optimized library.
Normalization: Wavefunctions satisfy quantum mechanical normalization.
Efficiency: Tridiagonal matrix diagonalization has O(N) storage.
'''