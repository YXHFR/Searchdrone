"""Gazebo's environment: the headless server keeps the GPU choice, only a window gets a display."""

import os

import pytest

from sd_sim import gazebo, procs

WSL_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/home/sd",
    "DISPLAY": ":0",
    "WAYLAND_DISPLAY": "wayland-0",
    "XDG_RUNTIME_DIR": "/run/user/1000",
    "PULSE_SERVER": "unix:/mnt/wslg/PulseServer",
    "GALLIUM_DRIVER": "d3d12",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
    "LIBGL_ALWAYS_SOFTWARE": "0",
    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib",
    "GZ_SIM_RESOURCE_PATH": "/models",
    "SECRET_TOKEN": "x",
}


@pytest.fixture
def wsl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        monkeypatch.delenv(key)
    for key, value in WSL_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.mark.usefixtures("wsl_env")
def test_headless_server_keeps_gpu_choice() -> None:
    env = gazebo.gz_env(0, gui=False)
    for key in procs.GPU_ENV_KEYS:
        assert env[key] == WSL_ENV[key]
    for key in procs.DISPLAY_ENV_KEYS:
        assert key not in env
    assert env["GZ_PARTITION"] == "sd_i0"
    assert env["GZ_SIM_RESOURCE_PATH"] == "/models"
    assert "SECRET_TOKEN" not in env


@pytest.mark.usefixtures("wsl_env")
def test_window_gets_display_and_gpu() -> None:
    env = gazebo.gz_env(3, gui=True)
    for key in procs.GRAPHICS_ENV_KEYS:
        assert env[key] == WSL_ENV[key]
    assert env["GZ_PARTITION"] == "sd_i3"
    assert "SECRET_TOKEN" not in env
