from mpmath import mp, mpf
from dataclasses import dataclass

@dataclass
class SolverParams:
    w0: float
    w1: float
    alpha_value: float
    tau: float
    num_neurons: int
    threshold: float
    external_current: float
    refractory_time: float

def solve_w(params: SolverParams):
    w0_mp = mpf(params.w0)
    w1_mp = mpf(params.w1)
    alpha_mp = mpf(params.alpha_value)
    tau_mp = mpf(params.tau)
    n_mp = mpf(params.num_neurons)
    theta_mp = mpf(params.threshold)
    i_mp = mpf(params.external_current)
    tau_ref_mp = mpf(params.refractory_time)

    def equation_w_mpmath(w_value):
        delta = (tau_mp * n_mp / (2 * i_mp)) * (
            (theta_mp - w_value * (n_mp - 1)) +
            mp.sqrt(
                (theta_mp - w_value * (n_mp - 1))**2 +
                (4 * i_mp * w_value * (n_mp - 1) * tau_ref_mp) / (n_mp * tau_mp)
            )
        )
        return w0_mp + (w1_mp - w0_mp) * mp.e**(-alpha_mp * delta) - w_value

    initial_guess = (
        theta_mp / (n_mp - 1)
        - 2 * i_mp * tau_ref_mp / (tau_mp * n_mp * (n_mp - 1))
    )

    solution = mp.findroot(equation_w_mpmath, initial_guess, solver="newton", tol=mp.mpf('1e-30'))
    residual = abs(equation_w_mpmath(solution))

    epsilon = mp.mpf('1e-10')
    derivative_val = abs(
        (equation_w_mpmath(solution + epsilon) - equation_w_mpmath(solution)) / epsilon
    )
    effective_derivative = max(derivative_val, mp.mpf('1e-10'))
    max_estimated_error = residual / effective_derivative

    if residual > mp.mpf('2e-10'):
        print("Warning: High residual detected:", residual)
    if derivative_val < mp.mpf('0.9'):
        print("Warning: Low derivative detected:", derivative_val)

    return solution, max_estimated_error

def compute_w_theoretical_crit(num_neurons, threshold, external_current, refractory_time, tau):
    return (
        threshold / (num_neurons - 1)
        - 2 * external_current * refractory_time
        / (tau * num_neurons * (num_neurons - 1))
    )
