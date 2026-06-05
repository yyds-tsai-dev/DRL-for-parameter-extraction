# IR-Drop Solver Continuation

The IR-drop current solve now uses multi-start continuation: previous accepted
Vds solution, zero-current start, then fixed-point warmup fallback. A solve is
accepted when SciPy reports `ier == 1` or the final residual is below
`IR_DROP_RESIDUAL_TOL`.

## Alternatives Considered

- Increase `IR_DROP_MAXFEV`: already set high enough that hard random-init cases
  still fail, because the starting point can enter the wrong basin.
- Use measured current as the solver start: converges well, but leaks target data
  into simulation and can make the RL objective less honest.
- Relax truncation for all non-converged solves: keeps more rollouts, but risks
  training on physically unreliable currents.

## Consequences

Training should produce fewer invalid rollouts without changing the NRMSE reward
or success definition. Solver diagnostics now include residual and attempt
metadata so future tuning can distinguish numerical failure from poor policy
quality.
