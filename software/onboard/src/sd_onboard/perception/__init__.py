"""Perception onboard: stills in, detections on the ground out, as reports to the ground.

``sources`` gives stills, ``pipeline`` turns one still and the pose at that moment into a
``PerceptionReport`` (the detector sources from the class register, georeferencing on
flat ground, small JPEG thumbnails of the targets) and ``loop`` runs it as its own task,
apart from the executor: a slow or failing detector never delays an order or the
emergency path, and it never steers.
"""
