"""The sim farm: a loop on the owner's computer that renders training pictures with facit.

``just sim-farm`` runs until it is stopped. Every round it brings a separate worktree
(``build/simfarm/main``) to the newest ``main``, syncs it, and runs one seeded search
round there: a random world, time of day, weather, flight height, people and decoys in
Gazebo, flown by the chosen commander (``commander=script`` or ``commander=sd-brain``).
The round writes its pictures with facit (YOLO and COCO boxes computed from the
simulator's ground truth and the camera's pose, plus one metadata JSON per picture),
checks itself, and the loop stores what passed: in the gitignored ``build/simfarm/``,
and with a key also in Cloudflare R2 with EU jurisdiction.

The parts:

* ``scenario`` draws a round's settings from a seed.
* ``people`` is the catalog of person models (CC0 / CC BY 4.0 from Gazebo Fuel, fetched
  by hash into ``build/``, never committed) and the person as a world object.
* ``camera`` and ``labels`` project the facit through the camera into picture boxes and
  write YOLO, COCO and the per-picture metadata.
* ``qc`` checks a round: the drone took off, the pictures are not black, the boxes agree
  with the simulator's bounding-box camera.
* ``store`` keeps rounds locally and uploads them to R2, resumable and never twice.
* ``host`` keeps the farm polite: it pauses while the GPU or CPU is busy with something
  else, keeps a duty cycle and a disk budget, and can wait until the computer is idle.
* ``loop`` is ``just sim-farm``; ``status`` is ``just sim-farm-status``.

No part of the farm recognises anyone: the only label is ``person``.
"""
