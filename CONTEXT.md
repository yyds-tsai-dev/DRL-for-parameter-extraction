# DRL Optimization

This context describes the language used to discuss reinforcement-learning based optimization problems in this repository, including EEHEMT model parameter extraction from measured current-voltage data and material composition optimization for predicted hardness.

## Language

**I-V Curve**:
A drain-current response measured or simulated across a gate-voltage sweep under one or more fixed curve conditions.
_Avoid_: curve, plot

**Curve Condition**:
A fixed bias or operating condition that identifies one measured or simulated I-V curve within a fitting dataset. One fitting dataset has many curve conditions.
_Avoid_: Vds-only condition, plot label

**Evaluation Curve**:
The official I-V curve snapshot used to inspect policy progress at an evaluation checkpoint. It is distinct from transient curves produced while collecting training experience.
_Avoid_: training curve, worker curve

**NRMSE Objective**:
The fitting goal for policy quality: minimise normalized root mean squared error in linear current space across the measured I-V Curve dataset. It is distinct from an arcsinh Huber fit, which may be used only if it is deliberately chosen as a surrogate for lowering NRMSE.
_Avoid_: arcsinh Huber objective, generic fit objective

**Material Composition**:
The alloy recipe evaluated by a hardness prediction model. It is expressed as element fraction columns rather than as an EEHEMT modelcard.
_Avoid_: model parameter, device parameter

**Tunable Composition Fraction**:
One of the material composition fractions controlled by the reinforcement-learning policy: Al, Cr, Mn, Fe, Co, or Ni.
_Avoid_: key parameter, simulator parameter

**Fixed Composition Fraction**:
A material composition fraction present in the hardness model input but intentionally held constant outside the reinforcement-learning action space. In the six-element hardness optimization setup, Cu and Mo are fixed at zero.
_Avoid_: ignored feature, missing feature

**Feasible Material Composition**:
A material composition whose Tunable Composition Fractions each stay within 0.05 and 0.35, whose six tunable fractions sum to 1.0, and whose Fixed Composition Fractions remain fixed.
_Avoid_: unconstrained fraction vector, raw action vector

**Hardness Objective**:
The material optimization goal for policy quality: find a Feasible Material Composition whose predicted hardness is at least 650. Model uncertainty may be inspected as a diagnostic, but it is not part of the first-pass objective.
_Avoid_: NRMSE Objective, fit objective

**Prediction Backend**:
The component that turns one candidate (an EEHEMT modelcard or a Material Composition) into predicted values, behind the `PredictionBackend` protocol. Physics simulators, committee model packages, and ANN surrogates are all Prediction Backends.
_Avoid_: the model (ambiguous), inference script

**Objective Strategy**:
The component that turns a predicted value into reward and success, and names the checkpoint-ranking metric. The NRMSE Objective and the Hardness Objective are realized by `NRMSEMinimizeObjective` and `ThresholdMaximizeObjective`. Episode control stays in the environment.
_Avoid_: reward function (partial), fitness

**Problem Spec**:
The registration record that binds one optimization problem's environment, training assembly, W&B project, and checkpoint metric under a `--env` name. Adding a problem means registering a Problem Spec, not editing the harness.
_Avoid_: env entry, config block

## Example Dialogue

Dev: "Should we save every I-V curve from parallel training?"

Domain expert: "No. The saved image should be the evaluation curve so the file represents one comparable checkpoint of parameter extraction progress."

Dev: "What does the legend separate?"

Domain expert: "Each line is grouped by curve condition, and each condition compares measured current against simulated current."

Dev: "Which metric decides whether a policy is better?"

Domain expert: "Use the NRMSE Objective. A lower arcsinh Huber loss is useful only when it also helps lower linear-current NRMSE."

Dev: "Should the alloy optimizer control Cu and Mo because the hardness model accepts them?"

Domain expert: "No. In the six-element setup, Cu and Mo are Fixed Composition Fractions. The policy controls only the Tunable Composition Fractions."

Dev: "Can a policy produce six fractions that add up to more than one?"

Domain expert: "No. A valid candidate must be a Feasible Material Composition."

Dev: "Should uncertainty reduce the reward when optimizing hardness?"

Domain expert: "Not in the first pass. Use predicted hardness for the Hardness Objective and keep uncertainty as a diagnostic."
