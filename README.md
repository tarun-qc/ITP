# Quantum Harmonic Oscillator: Imaginary Time Propagation and Hamiltonian Diagonalization

This repository contains a Python implementation for solving the one-dimensional quantum harmonic oscillator using two numerical approaches:

1. **Imaginary Time Propagation (ITP)** for obtaining the ground-state energy and wavefunction.
2. **Hamiltonian diagonalization** for obtaining multiple eigenvalues and eigenstates.

The numerical Hamiltonian is constructed using a three-point finite-difference approximation to the kinetic-energy operator. The resulting Hamiltonian is tridiagonal and is solved efficiently using `scipy.linalg.eigh_tridiagonal`.

## 1. Problem Description

The one-dimensional quantum harmonic oscillator is described by the Hamiltonian

\[
\hat{H} =
-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}
+ V(x),
\]

with the harmonic potential

\[
V(x) = \frac{1}{2}x^2.
\]

In this implementation,

\[
\hbar = 1, \qquad m = 1.
\]

The exact energy levels are

\[
E_n = n + \frac{1}{2},
\qquad n=0,1,2,\ldots
\]

The numerical results can therefore be compared against these analytical values.

## 2. Methods

### 2.1 Imaginary Time Propagation

Imaginary Time Propagation is used to obtain the ground state. The method starts from a random normalized wavefunction and repeatedly applies an imaginary-time evolution step:

\[
\psi_{\mathrm{new}}
=
\psi - \Delta t\,\hat{H}\psi.
\]

After every propagation step, the wavefunction is normalized. Repeated propagation suppresses higher-energy components and causes the wavefunction to converge toward the ground state.

The default parameters are:

```text
x_min = -10
x_max = 10
dx    = 0.05
dt    = 0.001
steps = 10000
```

The function returns the ground-state energy, wavefunction, spatial grid, and potential.

### 2.2 Hamiltonian Diagonalization

The Hamiltonian is discretized using a three-point finite-difference approximation:

\[
-\frac{d^2\psi}{dx^2}
\approx
-\frac{\psi_{i+1}-2\psi_i+\psi_{i-1}}{dx^2}.
\]

This produces a tridiagonal Hamiltonian matrix with diagonal elements

\[
H_{ii}
=
\frac{\hbar^2}{m\,dx^2}+V_i
\]

and off-diagonal elements

\[
H_{i,i+1}
=
H_{i+1,i}
=
-\frac{\hbar^2}{2m\,dx^2}.
\]

The lowest eigenvalues and eigenvectors are obtained using:

```python
scipy.linalg.eigh_tridiagonal
```

## 3. Requirements

The code requires Python 3 and the following packages:

- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
pip install numpy scipy matplotlib
```

## 4. Running the Code

Run the script with:

```bash
python ITP.py
```

The script first calculates the ground state using ITP:

```python
E0, psi0, x, V = imaginary_time_propagation()
```

It then calculates the first five states using Hamiltonian diagonalization:

```python
eigenvalues, eigenvectors, x, V = diagonalization_method(num_states=5)
```

The calculated energies are printed to the terminal.

## 5. Expected Results

The exact harmonic-oscillator energies are:

```text
E0 = 0.5
E1 = 1.5
E2 = 2.5
E3 = 3.5
E4 = 4.5
```

The script compares the numerical diagonalization results with these exact values.

The ITP calculation reports the numerical ground-state energy:

```text
Ground state energy (ITP): ...
```

Small differences between the numerical and analytical energies are expected because the calculation uses a finite spatial grid and finite-difference discretization.

## 6. Visualization

The script generates a plot showing:

- The harmonic oscillator potential.
- The ground-state eigenfunction.
- The first excited-state eigenfunction.
- The second excited-state eigenfunction.

The eigenfunctions are vertically shifted by their corresponding eigenvalues to visualize the energy levels.

## 7. Code Structure

```text
ITP.py
│
├── harmonic_potential()
│   └── Defines V(x) = 0.5 x²
│
├── imaginary_time_propagation()
│   ├── Creates spatial grid
│   ├── Constructs finite-difference kinetic operator
│   ├── Initializes random wavefunction
│   ├── Performs imaginary-time propagation
│   ├── Normalizes wavefunction
│   └── Calculates ground-state energy
│
├── diagonalization_method()
│   ├── Creates spatial grid
│   ├── Constructs tridiagonal Hamiltonian
│   ├── Diagonalizes Hamiltonian
│   └── Returns eigenvalues/eigenvectors
│
└── Main calculation
    ├── Runs ITP
    ├── Runs diagonalization
    ├── Compares numerical and exact energies
    └── Plots eigenstates
```

## 8. Numerical Considerations

### Spatial Grid

The default spatial domain is:

```python
x_min = -10
x_max = 10
dx = 0.05
```

A smaller `dx` gives a finer spatial representation but increases the number of grid points and computational cost.

### Imaginary-Time Step

The default timestep is:

```python
dt = 0.001
```

The timestep is important for numerical stability. A timestep that is too large can make the propagation unstable, whereas a timestep that is too small may require more iterations to achieve convergence.

### Number of Propagation Steps

The default number of steps is:

```python
steps = 10000
```

Increasing the number of steps can improve convergence toward the ground state.

## 9. Computational Efficiency

The Hamiltonian is tridiagonal because of the three-point finite-difference representation of the kinetic-energy operator.

The implementation therefore stores and operates on the diagonal and off-diagonal components rather than explicitly constructing a dense Hamiltonian matrix.

This makes the approach memory-efficient for larger one-dimensional grids.

## 10. Comparison of the Two Methods

| Method | Main Purpose | Output | Main Advantage |
|---|---|---|---|
| Imaginary Time Propagation | Ground state | Ground-state energy and wavefunction | Efficient for obtaining the ground state |
| Hamiltonian Diagonalization | Multiple states | Several energies and eigenstates | Direct access to multiple low-energy states |

ITP is particularly useful when only the ground state is required, while diagonalization is useful when several low-lying eigenstates are needed.

## 11. Physical Interpretation

If the initial wavefunction is expanded in terms of the Hamiltonian eigenstates,

\[
\psi(x,0)
=
\sum_n c_n\phi_n(x),
\]

imaginary-time evolution introduces factors proportional to

\[
e^{-E_n\tau}.
\]

Higher-energy states decay faster than the ground state. Consequently, after sufficient propagation,

\[
\psi(x,\tau)
\rightarrow
\phi_0(x),
\]

where \(\phi_0(x)\) is the ground-state wavefunction.

## 12. Possible Extensions

Possible future improvements include:

- Adding an explicit convergence criterion for ITP.
- Monitoring the energy during propagation.
- Comparing the numerical ground-state wavefunction with the analytical solution.
- Calculating numerical errors relative to the exact energies.
- Performing convergence tests with respect to `dx`, `dt`, and the number of propagation steps.
- Supporting arbitrary one-dimensional potentials.
- Implementing alternative propagation schemes such as Crank–Nicolson or split-operator propagation.
- Benchmarking ITP against direct diagonalization for different grid sizes.

## 13. License

This project is provided for research, educational, and computational experimentation purposes.
