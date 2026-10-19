"""``just perf``: measure that the foundation runs fast enough on this computer.

``stats`` holds the pure arithmetic (percentiles, window means, clock offsets),
``samplers`` collects RAM, VRAM, the Gazebo real-time factor and Windows memory while
the run goes on, ``tlogack`` finds the STOP acknowledgements in the radio tlog and
``report`` turns the measurements into P0-P8 plus the model check and writes
``data/reports/perf-<stamp>.json`` and ``.md``.
"""
