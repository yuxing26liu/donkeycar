Track 2: Agentic Development — New DonkeyCar Capabilities
Goal: Use Claude as a coding agent to build new autonomous behaviors, through an iterative "assign mission → agent writes code → test → give feedback → agent revises" loop.
Suggested mission progression (each stage can be its own project, or phases of one project):
Line following
Lane following (upgrade from a single line following to a full lane navigation)
Two-way road navigation — detect and avoid an oncoming robot approaching in the opposite lane, without a collision
Process: Give Claude a mission, have it write the control/vision code, test it on the car, then feed back what failed so it can revise its approach. Document how the agent's code evolved across iterations and what feedback drove each improvement.

Cross-Cutting Requirements (apply to every track)
Give Claude real access. The agent needs API-level access to the robot's controls (steering, throttle) and sensor streams (camera, IMU, and any added sensors) to meaningfully test and iterate — not just read code in isolation.
End-to-end sensor integration guide. Document, start to finish, how to connect a new sensor to DonkeyCar: wiring, drivers, software, and how it plugs into the control loop.
Configuration documentation. Clearly document how to configure each drive component — steering, throttle, camera, IMU — so future students (and future Claude sessions) can build on your work.
Documentation as a deliverable. Every project should produce documentation clear enough to be reused as reference material in future course offerings.
