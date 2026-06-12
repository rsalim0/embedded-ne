"""Thin MQTT publisher (paho-mqtt v2) for sending motor commands to the ESP8266."""
from __future__ import annotations

import threading

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(self, cfg):
        m = cfg["mqtt"]
        self.host = m["host"]
        self.port = int(m["port"])
        self.command_topic = m["command_topic"]
        self.status_topic = m["status_topic"]
        self.qos = int(m.get("qos", 0))
        self.keepalive = int(m.get("keepalive", 30))

        self._connected = threading.Event()
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=m.get("client_id_pc", "benax-pc"),
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    # -- lifecycle ---------------------------------------------------------- #
    def connect(self, timeout: float = 5.0) -> bool:
        self.client.connect_async(self.host, self.port, self.keepalive)
        self.client.loop_start()
        return self._connected.wait(timeout)

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    # -- publishing --------------------------------------------------------- #
    def publish_command(self, payload: str):
        """Publish a motor command token (e.g. 'RIGHT:4') to the command topic."""
        self.client.publish(self.command_topic, payload, qos=self.qos)

    def publish_status(self, payload: str):
        self.client.publish(self.status_topic, payload, qos=self.qos)

    # -- callbacks ---------------------------------------------------------- #
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._connected.set()
            print(f"[mqtt] connected to {self.host}:{self.port}")
        else:
            print(f"[mqtt] connect failed: {reason_code}")

    def _on_disconnect(self, client, userdata, *args):
        self._connected.clear()
        print("[mqtt] disconnected")
