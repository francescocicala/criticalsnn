from mpmath import mp, mpf
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SolverParams:
    """
    Data class holding the parameters for the solver of the synaptic weight equation.
    
    Attributes:
        w0 (float): Initial lower-bound weight parameter.
        w1 (float): Initial upper-bound weight parameter.
        alpha_value (float): Alpha parameter controlling the decay rate in the exponent.
        tau (float): Time interval or timescale parameter used in the equation.
        num_neurons (int): Number of neurons in the network.
        threshold (float): Firing threshold.
        external_current (float): External current supplied to neurons.
        refractory_time (float): Neuronal refractory period.
    """
    w0: float
    w1: float
    alpha_value: float
    tau: float
    num_neurons: int
    threshold: float
    external_current: float
    refractory_time: float

def solve_w(params: SolverParams) -> Tuple[mpf, mpf]:
    """
    Solve for the synaptic weight 'w' using a self-consistency equation and MPMath's root-finding algorithm.
    
    The function defines a nonlinear equation for the synaptic weight 'w' based on the parameters in
    `SolverParams`, and then uses MPMath to find a numerical solution. It also computes an estimate of the
    maximum error by examining the equation's residual and its local derivative.

    Args:
        params (SolverParams): The solver parameters (w0, w1, alpha_value, tau, num_neurons, threshold,
                               external_current, refractory_time).

    Returns:
        Tuple[mpf, mpf]:
            - The solution for 'w' (as an mpmath mpf).
            - The estimated maximum error of the solution (as an mpmath mpf).
    """
    w0_mp = mpf(params.w0)
    w1_mp = mpf(params.w1)
    alpha_mp = mpf(params.alpha_value)
    tau_mp = mpf(params.tau)
    n_mp = mpf(params.num_neurons)
    theta_mp = mpf(params.threshold)
    i_mp = mpf(params.external_current)
    tau_ref_mp = mpf(params.refractory_time)

    def equation_w_mpmath(w_value: mpf) -> mpf:
        """
        Define the self-consistency equation for synaptic weight 'w'.
        """
        delta = (tau_mp * n_mp / (2 * i_mp)) * (
            (theta_mp - w_value * (n_mp - 1))
            + mp.sqrt(
                (theta_mp - w_value * (n_mp - 1))**2 +
                (4 * i_mp * w_value * (n_mp - 1) * tau_ref_mp) / (n_mp * tau_mp)
            )
        )
        return w0_mp + (w1_mp - w0_mp) * mp.e**(-alpha_mp * delta) - w_value

    # A heuristic initial guess for the root
    initial_guess = (
        theta_mp / (n_mp - 1)
        - (2 * i_mp * tau_ref_mp) / (tau_mp * n_mp * (n_mp - 1))
    )

    # Find the root of the equation
    solution = mp.findroot(equation_w_mpmath, initial_guess, solver="newton", tol=mp.mpf('1e-30'))
    residual = abs(equation_w_mpmath(solution))

    # Compute derivative-based error estimate
    epsilon = mp.mpf('1e-10')
    derivative_val = abs(
        (equation_w_mpmath(solution + epsilon) - equation_w_mpmath(solution)) / epsilon
    )
    effective_derivative = max(derivative_val, mp.mpf('1e-10'))
    max_estimated_error = residual / effective_derivative

    # Print warnings if solution quality might be compromised
    if residual > mp.mpf('2e-10'):
        print("Warning: High residual detected:", residual)
    if derivative_val < mp.mpf('0.9'):
        print("Warning: Low derivative detected:", derivative_val)

    return solution, max_estimated_error

def compute_w_theoretical_crit(num_neurons: int,
                               threshold: float,
                               external_current: float,
                               refractory_time: float,
                               tau: float) -> float:
    """
    Compute the theoretical critical value for 'w' given model parameters.
    
    Args:
        num_neurons (int): Number of neurons in the network.
        threshold (float): Firing threshold of the neurons.
        external_current (float): External current supplied to neurons.
        refractory_time (float): Neuronal refractory period.
        tau (float): Time interval or timescale parameter used in the analysis.
    
    Returns:
        float: The theoretical critical synaptic weight, computed via a derived formula.
    """
    return float(
        threshold / (num_neurons - 1)
        - (2 * external_current * refractory_time)
        / (tau * num_neurons * (num_neurons - 1))
    )
