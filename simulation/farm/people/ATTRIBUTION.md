# Person models of the sim farm

The sim farm places these models from [Gazebo Fuel](https://app.gazebosim.org) in its
worlds. They are not in this repository: `just sim-farm-setup` fetches each pinned
version into `build/simfarm/assets/people/` and checks its SHA-256. The files next to
this one (`<id>.toml`) record where each comes from, its licence and its measured box.
Every picture the farm makes lists the models in its round with these credits.

| Model | Credit | Licence | Source |
|---|---|---|---|
| Standing person v3 | Marina Kollmitz, Open Robotics, made with MakeHuman | CC0 1.0 | https://app.gazebosim.org/OpenRobotics/fuel/models/Standing%20person |
| Walking person v3 | Marina Kollmitz, Open Robotics, made with MakeHuman | CC0 1.0 | https://app.gazebosim.org/OpenRobotics/fuel/models/Walking%20person |
| Casual female v4 | Rohit Salem, Open Robotics, made with MakeHuman | CC0 1.0 | https://app.gazebosim.org/OpenRobotics/fuel/models/Casual%20female |
| Rescue Randy v3 | Open Robotics (Ian Chen, Cole Biesemeyer) | CC BY 4.0 | https://app.gazebosim.org/OpenRobotics/fuel/models/Rescue%20Randy |
| Rescue Randy Sitting v8 | Open Robotics (Ian Chen, Cole Biesemeyer) | CC0 1.0 | https://app.gazebosim.org/OpenRobotics/fuel/models/Rescue%20Randy%20Sitting |

Rescue Randy is used under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The farm changes it only by placing it,
turning it (lying) or giving it one solid colour. The other models are CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/). The catalog files here are the
project's own (Apache-2.0); the models' licences apply to the models only.

The farm only labels "person". It never recognises who a person is, and the models'
faces are never labelled.
