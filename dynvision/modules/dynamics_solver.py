"""
Modules to stepwise solve the dynamical systems differential equation:
$$$
\tau \frac{dx}{dt} = -x + \frac{dt}{tau} * [-x + W(x)]
$$$
to evolve the activity tensor from x(t) to x(t+dt) with a given time step dt and time constant tau.
"""

from typing import Union

import torch
import torch.nn as nn

__all__ = ["EulerStep", "RungeKuttaStep"]


class EulerStep(nn.Module):
    """
    $$$
    x(t+dt) = x(t) + dt/tau * [-x(t) + W(x(t))]
    $$$
    implemented with the variable names: x(t+dt) = y, x(t) = h, and W(x(t)) = x
    """

    def __init__(
        self, dt: float = 2.0, tau: Union[float, torch.nn.Parameter] = 5.0, device=None
    ) -> None:

        super(EulerStep, self).__init__()

        self.dt = torch.tensor(dt, requires_grad=False, device=device)

        if isinstance(tau, (int, float)):
            self.tau = torch.tensor(tau, requires_grad=False, device=device)
        else:
            # pass parameterized tau if it should be trainable
            self.tau = tau

        self.reset()

    def reset(self):
        # hidden state has a single slot
        self.hidden_state = None

    def forward(
        self, x: Union[torch.Tensor, None], h: Union[torch.Tensor, None] = None
    ) -> Union[torch.Tensor, None]:

        if x is None:
            return None

        use_own_hidden_state = h is None

        if use_own_hidden_state and self.hidden_state is not None:
            h = self.hidden_state.to(x.device)

        dt = self.dt.to(x.device)
        tau = self.tau.to(x.device)

        if h is None:
            y = dt / tau * x
        else:
            y = h + dt / tau * (x - h)

        if use_own_hidden_state:
            self.hidden_state = y

        return y


class RungeKuttaStep(nn.Module):
    """
    $$$
    x(t+dt) = x(t) + \frac{dt}{6\tau} * (k1 + 2*k2 + 2*k3 + k4)
    $$$
    where:
    $$$
    k1 = dt/tau * (x(t) - h)
    k2 = dt/tau * (x(t) - (h + 0.5 * k1))
    k3 = dt/tau * (x(t) - (h + 0.5 * k2))
    k4 = dt/tau * (x(t) - (h + k3))
    $$$
    implemented with the variable names: x(t+dt) = y, x(t) = h, and W(x(t)) = x
    """

    def __init__(
        self, dt: float = 2.0, tau: Union[float, torch.nn.Parameter] = 5.0
    ) -> None:

        super(RungeKuttaStep, self).__init__()

        self.dt = torch.tensor(dt, requires_grad=False)

        if isinstance(tau, (int, float)):
            self.tau = torch.tensor(tau, requires_grad=False)
        else:
            # pass parameterized tau if it should be trainable
            self.tau = tau

        self.reset()

    def reset(self):
        # hidden state has a single slot
        self.hidden_state = None

    def forward(
        self, x: Union[torch.Tensor, None], h: Union[torch.Tensor, None] = None
    ) -> Union[torch.Tensor, None]:

        if x is None:
            return None

        use_own_hidden_state = h is None

        if use_own_hidden_state and self.hidden_state is not None:
            h = self.hidden_state.to(x.device)

        dt = self.dt.to(x.device)
        tau = self.tau.to(x.device)

        if h is None:
            h = torch.zeros_like(x)

        k1 = dt / tau * (x - h)
        k2 = dt / tau * (x - (h + 0.5 * k1))
        k3 = dt / tau * (x - (h + 0.5 * k2))
        k4 = dt / tau * (x - (h + k3))

        y = h + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if use_own_hidden_state:
            self.hidden_state = y

        return y


if __name__ == "__main__":
    dt = 0.1
    tau = 1.0
    inf = 1000
    rtol = 1e-03
    atol = 1e-04

    for solver_name in __all__:
        solver = globals()[solver_name]
        solver_step = solver(dt, tau)
        print(f"Testing {solver_name} with dt={dt}, tau={tau}, inf={inf}:")

        # System with constant input approaches the input value
        c = torch.tensor([1.0])
        print(f"\t System with constant input c={c.item()}:")
        for t in range(inf):
            y = solver_step(c)

        print(f"\t W(x)={c.item()}), \t t->inf, \t y=", y.item())
        assert torch.allclose(y, c, rtol=rtol, atol=atol)

        solver_step.reset()

        # Exponential decay (k<1)
        k = 0.7
        print(f"\t Exponential decay (k={k}<1):")
        x_0 = torch.tensor([1.0])
        x_1 = torch.tensor([0.8])
        y = torch.tensor([0.0])
        x = solver_step(x_1, x_0)
        for t in range(inf):
            x = solver_step(k * x)

        print(f"\t W(x)={k}x), \t t->inf, \t x=", x.item())
        assert torch.allclose(x, y, rtol=rtol, atol=atol)

        solver_step.reset()

        # None input
        assert solver_step(None) is None

    print("All tests passed!")
