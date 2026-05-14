import json
import threading
import time

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False


class ShellyMqttTrigger:
    """Envoie une impulsion MQTT au Shelly 1 Mini Gen 4 via RPC MQTT (Gen 2 API).

    Le relais s'active a la detection d'une plaque autorisee, puis se coupe
    automatiquement apres `pulse_duration_sec` secondes.
    """

    def __init__(self, broker_host, broker_port, username, password,
                 shelly_device_id, pulse_duration_sec=3.0):
        if not _PAHO_AVAILABLE:
            raise RuntimeError(
                "paho-mqtt n'est pas installe. Lance: pip install paho-mqtt"
            )

        self._broker_host = broker_host
        self._broker_port = int(broker_port)
        self._username = username
        self._password = password
        self._device_id = shelly_device_id
        self._pulse_duration_sec = float(pulse_duration_sec)

        # Topics RPC Shelly Gen 2/3/4
        self._rpc_topic = f"{shelly_device_id}/rpc"

        self._client = None
        self._connected = False
        self._lock = threading.Lock()
        self._pulse_timer = None
        self._msg_id = 0

        self._connect()

    def _connect(self):
        try:
            client = mqtt.Client(
                client_id="ai_camera_plate_detector",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )
            client.username_pw_set(self._username, self._password)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.connect_async(self._broker_host, self._broker_port, keepalive=60)
            client.loop_start()
            self._client = client
            print(
                f"[MQTT] Connexion au broker {self._broker_host}:{self._broker_port}..."
            )
        except Exception as exc:
            print(f"[MQTT] Erreur de connexion: {exc}")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._connected = True
            print(f"[MQTT] Connecte. Topic Shelly: {self._rpc_topic}")
        else:
            print(f"[MQTT] Echec connexion, code: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self._connected = False
        if reason_code != 0:
            print(f"[MQTT] Deconnexion inattendue (code {reason_code}), reconnexion auto...")

    def _publish_switch(self, on: bool):
        if self._client is None or not self._connected:
            print("[MQTT] Non connecte, commande ignoree.")
            return False

        with self._lock:
            self._msg_id += 1
            msg_id = self._msg_id

        payload = json.dumps({
            "id": msg_id,
            "src": "ai_camera",
            "method": "Switch.Set",
            "params": {"id": 0, "on": on},
        })

        try:
            result = self._client.publish(self._rpc_topic, payload, qos=1)
            state = "ON" if on else "OFF"
            if result.rc == 0:
                print(f"[MQTT] Shelly {self._device_id} -> {state}")
                return True
            else:
                print(f"[MQTT] Echec publication (rc={result.rc})")
                return False
        except Exception as exc:
            print(f"[MQTT] Erreur publication: {exc}")
            return False

    def _schedule_off(self):
        if self._pulse_timer is not None:
            self._pulse_timer.cancel()

        def _turn_off():
            self._publish_switch(False)

        self._pulse_timer = threading.Timer(self._pulse_duration_sec, _turn_off)
        self._pulse_timer.daemon = True
        self._pulse_timer.start()

    def trigger(self, plate: str):
        """Active le relais Shelly et programme son extinction automatique."""
        print(f"[MQTT] Plaque autorisee detectee: {plate} -> activation Shelly")
        if self._publish_switch(True):
            self._schedule_off()

    def disconnect(self):
        if self._pulse_timer is not None:
            self._pulse_timer.cancel()
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._connected = False
        print("[MQTT] Deconnecte.")
