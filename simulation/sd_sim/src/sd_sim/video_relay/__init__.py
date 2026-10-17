"""Video for the simulator (instance 0): the synthetic camera and the relay to MediaMTX.

``python -m sd_sim.video_relay source --port <video_raw_udp>`` sends the synthetic
picture with the machine-readable time strip (:mod:`.timecode`) as H.264 RTP, the same
way the Gazebo camera does. ``python -m sd_sim.video_relay relay`` takes the RTP that
``linksim`` passes on (``video_out_udp``) and publishes it to
``rtsp://127.0.0.1:8554/drone/<vehicle_id>/main`` with the password from a file.
"""

from sd_sim.video_relay.rtsp import RtspPublisher, stream_url
from sd_sim.video_relay.timecode import decode_luma, strip_row

__all__ = ["RtspPublisher", "decode_luma", "stream_url", "strip_row"]
