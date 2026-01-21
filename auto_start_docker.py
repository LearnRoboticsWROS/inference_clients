#!/usr/bin/env python3
import subprocess
import sys

# CONFIG
CONTAINER_NAME = "myserver"
IMAGE_NAME = "myimage"
PORT_MAPPING   = "9001:9001"  # adjust your ports

def is_container_running(name):
    """Check if the container is already running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        running_containers = result.stdout.strip().split("\n")
        return name in running_containers
    except Exception as e:
        print(f"Error checking container: {e}", file=sys.stderr)
        return False

def start_container(name, image, port):
    """Start the container if not running"""
    try:
        print(f"Starting container {name}...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", name,
            "-p", port,
            "--restart", "unless-stopped",
            image
        ])
        print("Container started!")
    except Exception as e:
        print(f"Error starting container: {e}", file=sys.stderr)

if __name__ == "__main__":
    if not is_container_running(CONTAINER_NAME):
        start_container(CONTAINER_NAME, IMAGE_NAME, PORT_MAPPING)
    else:
        print(f"Container {CONTAINER_NAME} is already running.")

