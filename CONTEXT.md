# DRL Optimization

This context describes the language used to discuss reinforcement-learning based optimization problems in this repository, including EEHEMT parameter extraction from measured current-voltage data and material composition optimization for predicted hardness.

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

**Episode-Best NRMSE**:
The lowest NRMSE Objective value reached within a single policy episode. It is distinct from the final NRMSE at the episode's last step.
_Avoid_: final NRMSE, last NRMSE

**Material Composition**:
The alloy recipe evaluated by a hardness prediction model. It is expressed as element fraction columns rather than as an EEHEMT modelcard.
_Avoid_: model parameter, device parameter

**Tunable Composition Fraction**:
One of the material composition fractions controlled by the reinforcement-learning policy: Al, Cr, Mn, Fe, Co, or Ni.
_Avoid_: key parameter, simulator parameter

**Fixed Composition Fraction**:
A material composition fraction present in the hardness model input but intentionally held constant outside the reinforcement-learning action space, such as Cu or Mo in the six-element optimization setup.
_Avoid_: ignored feature, missing feature

**Hardness Objective**:
The material optimization goal for policy quality: find a material composition whose predicted hardness exceeds the target threshold.
_Avoid_: NRMSE Objective, fit objective

## Example Dialogue

Dev: "Should we save every I-V curve from parallel training?"

Domain expert: "No. The saved image should be the evaluation curve so the file represents one comparable checkpoint of parameter extraction progress."

Dev: "What does the legend separate?"

Domain expert: "Each line is grouped by curve condition, and each condition compares measured current against simulated current."

Dev: "Which metric decides whether a policy is better?"

Domain expert: "Use the NRMSE Objective. A lower arcsinh Huber loss is useful only when it also helps lower linear-current NRMSE."

Dev: "Should the alloy optimizer control Cu and Mo because the hardness model accepts them?"

Domain expert: "No. In the six-element setup, Cu and Mo are Fixed Composition Fractions. The policy controls only the Tunable Composition Fractions."
