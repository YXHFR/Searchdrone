# ArduPilot SITL (ArduCopter) image for headless simulator tests.
# Build from the repository root:
#   docker build -f simulation/docker/sitl.Dockerfile -t searchdrone-sitl .
# Versions come from simulation/versions.lock; the base image default below must equal
# its ubuntu_image (a unit test checks it). Only simulation/docker/ and
# simulation/versions.lock may be copied in, so the image tag (a hash of exactly those
# files) changes whenever the image can change.

ARG UBUNTU_IMAGE=ubuntu:noble-20260911@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

# --- 1. build ArduCopter SITL ----------------------------------------------------------
FROM ${UBUNTU_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git sudo python3 lsb-release tzdata \
 && rm -rf /var/lib/apt/lists/*
# ArduPilot's prerequisite script must not run as root; like ArduPilot's own dev image,
# a build user with password-less sudo runs it.
RUN useradd --create-home --shell /bin/bash builder \
 && echo "builder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/builder \
 && chmod 0440 /etc/sudoers.d/builder
COPY simulation/versions.lock /tmp/versions.lock
COPY simulation/docker/known_params.py /tmp/known_params.py
USER builder
WORKDIR /home/builder
RUN set -eu; \
    ver() { python3 -c "import tomllib,sys; print(tomllib.load(open('/tmp/versions.lock','rb'))[sys.argv[1]])" "$1"; }; \
    tag="$(ver ardupilot_tag)"; commit="$(ver ardupilot_commit)"; \
    git clone --branch "$tag" --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/ArduPilot/ardupilot.git ardupilot; \
    cd ardupilot; \
    [ "$(git rev-parse HEAD)" = "$commit" ] || { echo "wrong ArduPilot commit"; exit 1; }; \
    echo "$tag" > .sd_version
WORKDIR /home/builder/ardupilot
RUN USER=builder DO_AP_STM_ENV=0 DO_PYTHON_VENV_ENV=0 SKIP_AP_GRAPHIC_ENV=1 SKIP_AP_COV_ENV=1 \
    SKIP_AP_COMPLETION_ENV=1 SKIP_AP_GIT_CHECK=1 \
    Tools/environment_install/install-prereqs-ubuntu.sh -y \
 && sudo rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    export PATH="/home/builder/venv-ardupilot/bin:$PATH"; \
    ./waf configure --board sitl; \
    ./waf copter; \
    python3 Tools/autotest/param_metadata/param_parse.py --vehicle ArduCopter --format xml; \
    python3 /tmp/known_params.py apm.pdef.xml > known_params.txt

# --- 2. runtime ------------------------------------------------------------------------
FROM ${UBUNTU_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/YXHFR/Searchdrone" \
      org.opencontainers.image.licenses="GPL-3.0-or-later AND Apache-2.0" \
      org.opencontainers.image.description="ArduPilot SITL (ArduCopter) for Searchdrone simulator tests"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git python3 python3-venv iproute2 postgresql-client \
      gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-rtsp \
      python3-gi gir1.2-gstreamer-1.0 \
 && rm -rf /var/lib/apt/lists/*
# Docker CLI from Docker's signed apt repository (for tests with their own database).
RUN install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
 && test "$(gpg --show-keys --with-colons /etc/apt/keyrings/docker.asc \
      | awk -F: '$1 == "pub" {w = 1; next} w && $1 == "fpr" {print $10; w = 0}')" \
      = 9DC858229FC7DD38854AE2D88D81803C0EBFCD88 \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*
COPY simulation/versions.lock /tmp/versions.lock
# uv: the release tarball, checked against the locked checksum before unpacking.
RUN set -eu; \
    ver() { python3 -c "import tomllib,sys; print(tomllib.load(open('/tmp/versions.lock','rb'))[sys.argv[1]])" "$1"; }; \
    uv_version="$(ver uv_version)"; uv_sha256="$(ver uv_sha256)"; \
    tgz=uv-x86_64-unknown-linux-gnu.tar.gz; tmp="$(mktemp -d)"; \
    curl -fsSL -o "$tmp/$tgz" "https://github.com/astral-sh/uv/releases/download/${uv_version}/${tgz}"; \
    echo "${uv_sha256}  $tmp/$tgz" | sha256sum -c -; \
    tar -xzf "$tmp/$tgz" -C "$tmp"; \
    install -m 0755 "$tmp/uv-x86_64-unknown-linux-gnu/uv" "$tmp/uv-x86_64-unknown-linux-gnu/uvx" /usr/local/bin/; \
    rm -rf "$tmp" /tmp/versions.lock
COPY --from=build /home/builder/ardupilot/build/sitl/bin/arducopter /opt/ardupilot/build/sitl/bin/arducopter
COPY --from=build /home/builder/ardupilot/Tools/autotest/default_params /opt/ardupilot/Tools/autotest/default_params
COPY --from=build /home/builder/ardupilot/known_params.txt /opt/ardupilot/known_params.txt
COPY --from=build /home/builder/ardupilot/.sd_version /opt/ardupilot/.sd_version
RUN useradd --create-home --shell /bin/bash sim
ENV ARDUPILOT_HOME=/opt/ardupilot \
    SD_ARDUCOPTER_BIN=/opt/ardupilot/build/sitl/bin/arducopter \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_PYTHON_DOWNLOADS=never
USER sim
WORKDIR /home/sim
