# DRL Parameter Extraction

This context describes the language used to discuss reinforcement-learning based extraction of EEHEMT model parameters from measured current-voltage data.

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

## Example Dialogue

Dev: "Should we save every I-V curve from parallel training?"

Domain expert: "No. The saved image should be the evaluation curve so the file represents one comparable checkpoint of parameter extraction progress."

Dev: "What does the legend separate?"

Domain expert: "Each line is grouped by curve condition, and each condition compares measured current against simulated current."

Dev: "Which metric decides whether a policy is better?"

Domain expert: "Use the NRMSE Objective. A lower arcsinh Huber loss is useful only when it also helps lower linear-current NRMSE."
