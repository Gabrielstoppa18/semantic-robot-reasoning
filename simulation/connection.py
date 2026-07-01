"""Connection helper for CoppeliaSim's ZeroMQ remote API.

Requires CoppeliaSim to be running (the ZMQ remote API server add-on
starts automatically with it) and the `coppeliasim-zmqremoteapi-client`
package installed (see requirements.txt).
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def connect():
    """Return the `sim` API object for a running CoppeliaSim instance."""
    client = RemoteAPIClient()
    return client.require("sim")


if __name__ == "__main__":
    sim = connect()
    print("Connected. Simulation time:", sim.getSimulationTime())
