/*
 * esp8266_servo.ino
 * -----------------------------------------------------------------------------
 * Camera-tracking servo controller for ESP8266 (NodeMCU / Wemos D1 mini),
 * driven over MQTT.
 *
 * The ESP8266 subscribes to a command topic and moves a single servo smoothly
 * to follow LEFT / RIGHT / CENTER / STOP commands, or sweeps autonomously in
 * SCAN mode. It publishes heartbeats / acks to a status topic.
 *
 * -----------------------------------------------------------------------------
 * WIRING / POWER (read this!):
 *   - Servo signal  -> D4 (GPIO2)
 *   - Servo V+      -> EXTERNAL 5V supply (NOT the ESP8266 3V3 pin!)
 *   - Servo GND     -> external supply GND  *AND*  ESP8266 GND  (common ground)
 *
 *   The servo MUST share GND with the ESP8266, and it MUST be powered from a
 *   separate 5V source (e.g. into VIN/5V from a 5V/2A adapter, or its own BEC).
 *   The 3V3 regulator on these boards cannot source the stall/inrush current a
 *   servo needs; powering a servo from 3V3 will brown-out and reset the chip.
 *   The firmware below only drives the PWM signal line.
 *
 * -----------------------------------------------------------------------------
 * Libraries (Arduino Library Manager / ESP8266 core):
 *   - ESP8266WiFi.h   (ESP8266 Arduino core)
 *   - PubSubClient.h  (Nick O'Leary, MQTT client)
 *   - Servo.h         (ESP8266 core's Servo implementation)
 *
 * MQTT command protocol (ASCII payload on COMMAND_TOPIC):
 *   "RIGHT:<n>"  -> targetAngle += n   (n = 1..10 deg, default 3 if omitted)
 *   "LEFT:<n>"   -> targetAngle -= n   (n = 1..10 deg, default 3 if omitted)
 *   "CENTER"     -> targetAngle = 90
 *   "STOP"       -> targetAngle = currentAngle (hold here)
 *   "SCAN"       -> autonomous sweep between SCAN_MIN..SCAN_MAX until any
 *                   other command arrives (which exits scan mode)
 *   Payloads are trimmed of whitespace and parsed case-insensitively-ish
 *   (commands are expected uppercase but leading/trailing spaces are tolerated).
 * -----------------------------------------------------------------------------
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

/* ===========================================================================
 * CONFIG  --  EDIT THESE
 * ======================================================================== */
#define WIFI_SSID      "Mr.dnd"
#define WIFI_PASS      "mrsdnd975"

#define MQTT_HOST      "192.168.1.100"   // PC running Mosquitto (Ethernet LAN IP)
#define MQTT_PORT      1884              // 1884: winget Mosquitto service holds 1883 (localhost-only)

#define COMMAND_TOPIC  "benax/camera/command"   // we SUBSCRIBE here
#define STATUS_TOPIC   "benax/camera/status"    // we PUBLISH heartbeats/acks here

#define CLIENT_ID_BASE "benax-esp8266"          // chip id is appended for uniqueness

/* Hardware */
#define SERVO_PIN      D4                // GPIO2 -- servo signal line (matches wiring)
                                          // NOTE: D4/GPIO2 is a boot pin (must be HIGH at
                                          // boot) and the onboard LED. Servo input is
                                          // high-impedance so boot is fine; if the board
                                          // ever fails to boot with the servo attached,
                                          // move the signal wire to D5/D6 and update this.

/* Servo motion tuning */
static const float    ANGLE_MIN      = 0.0f;
static const float    ANGLE_MAX      = 180.0f;
static const float    ANGLE_CENTER   = 90.0f;
static const float    EASE_STEP_DEG  = 1.0f;    // max change per motion tick (smoothness)
static const uint32_t MOTION_PERIOD  = 15;      // ms between motion ticks

/* Default step when "LEFT"/"RIGHT" arrive without ":<n>" */
static const int      DEFAULT_STEP   = 3;
static const int      STEP_MIN       = 1;
static const int      STEP_MAX       = 10;

/* SCAN mode sweep limits and speed */
static const float    SCAN_MIN       = 30.0f;
static const float    SCAN_MAX       = 150.0f;
static const float    SCAN_STEP_DEG  = 0.5f;    // how far targetAngle moves per scan tick
static const uint32_t SCAN_PERIOD    = 20;      // ms between scan target updates

/* Connectivity timing */
static const uint32_t HEARTBEAT_PERIOD = 3000;  // ms between status heartbeats
static const uint32_t MQTT_RETRY_PERIOD = 3000; // ms between MQTT reconnect attempts

/* Incoming payload buffer (PubSubClient gives bytes+length, NOT null-terminated) */
static const size_t   PAYLOAD_BUF_SIZE = 64;

/* ===========================================================================
 * GLOBAL STATE
 * ======================================================================== */
WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

Servo servo;

float targetAngle  = ANGLE_CENTER;   // where we want the servo to be
float currentAngle = ANGLE_CENTER;   // where the servo currently is (eased)

bool  scanMode     = false;          // autonomous sweep active?
int   scanDir      = +1;             // current scan sweep direction (+1 / -1)

char  clientId[40];                  // CLIENT_ID_BASE + chip id

uint32_t lastMotionMs    = 0;
uint32_t lastScanMs      = 0;
uint32_t lastHeartbeatMs = 0;
uint32_t lastMqttTryMs   = 0;

/* ===========================================================================
 * HELPERS
 * ======================================================================== */

/* Clamp a float into [ANGLE_MIN, ANGLE_MAX]. */
static float clampAngle(float a) {
  if (a < ANGLE_MIN) return ANGLE_MIN;
  if (a > ANGLE_MAX) return ANGLE_MAX;
  return a;
}

/* Publish a small JSON-ish status string to STATUS_TOPIC. */
static void publishStatus(const char *state) {
  if (!mqtt.connected()) return;
  char msg[80];
  // Keep it simple: {"angle":90,"mode":"track","state":"hb"}
  snprintf(msg, sizeof(msg),
           "{\"angle\":%d,\"mode\":\"%s\",\"state\":\"%s\"}",
           (int)lround(currentAngle),
           scanMode ? "scan" : "track",
           state);
  mqtt.publish(STATUS_TOPIC, msg);
}

/* ===========================================================================
 * COMMAND PARSING
 * ======================================================================== */

/* Trim leading/trailing whitespace in-place. */
static void trimInPlace(char *s) {
  // leading
  char *start = s;
  while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') start++;
  if (start != s) memmove(s, start, strlen(start) + 1);
  // trailing
  size_t len = strlen(s);
  while (len > 0) {
    char c = s[len - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      s[--len] = '\0';
    } else {
      break;
    }
  }
}

/*
 * Parse the step value following a "LEFT"/"RIGHT" command.
 * `arg` points at the text after the ':' (may be NULL or empty).
 * Returns a clamped step in [STEP_MIN, STEP_MAX], or DEFAULT_STEP if absent.
 */
static int parseStep(const char *arg) {
  if (arg == NULL || *arg == '\0') return DEFAULT_STEP;
  int n = atoi(arg);
  if (n < STEP_MIN) n = STEP_MIN;
  if (n > STEP_MAX) n = STEP_MAX;
  return n;
}

/* Any explicit motion command cancels SCAN mode. */
static void exitScanIfNeeded(const char *reason) {
  if (scanMode) {
    scanMode = false;
    Serial.printf("[CMD] SCAN cancelled by %s\n", reason);
  }
}

/* Handle a fully-formed, trimmed command string. */
static void handleCommand(char *cmd) {
  Serial.printf("[CMD] received: '%s'\n", cmd);

  // Split off the optional ":<n>" argument.
  char *colon = strchr(cmd, ':');
  char *arg   = NULL;
  if (colon != NULL) {
    *colon = '\0';      // terminate the verb
    arg = colon + 1;    // argument string
  }

  if (strcmp(cmd, "RIGHT") == 0) {
    exitScanIfNeeded("RIGHT");
    int step = parseStep(arg);
    targetAngle = clampAngle(targetAngle + step);
    Serial.printf("[CMD] RIGHT %d -> target=%.1f\n", step, targetAngle);
    publishStatus("ack");

  } else if (strcmp(cmd, "LEFT") == 0) {
    exitScanIfNeeded("LEFT");
    int step = parseStep(arg);
    targetAngle = clampAngle(targetAngle - step);
    Serial.printf("[CMD] LEFT %d -> target=%.1f\n", step, targetAngle);
    publishStatus("ack");

  } else if (strcmp(cmd, "CENTER") == 0) {
    exitScanIfNeeded("CENTER");
    targetAngle = ANGLE_CENTER;
    Serial.println("[CMD] CENTER -> target=90");
    publishStatus("ack");

  } else if (strcmp(cmd, "STOP") == 0) {
    exitScanIfNeeded("STOP");
    targetAngle = currentAngle;   // hold exactly where we are
    Serial.printf("[CMD] STOP -> hold at %.1f\n", targetAngle);
    publishStatus("ack");

  } else if (strcmp(cmd, "SCAN") == 0) {
    if (!scanMode) {
      scanMode = true;
      // Seed the sweep direction toward the nearer end of the scan range.
      scanDir = (targetAngle < (SCAN_MIN + SCAN_MAX) * 0.5f) ? +1 : -1;
      Serial.println("[CMD] SCAN -> entering autonomous sweep");
    }
    publishStatus("ack");

  } else {
    Serial.printf("[CMD] unknown command: '%s'\n", cmd);
    publishStatus("nack");
  }
}

/* ===========================================================================
 * MQTT CALLBACK
 * ======================================================================== */
static void mqttCallback(char *topic, byte *payload, unsigned int length) {
  // Copy into a fixed-size, null-terminated buffer (payload is NOT terminated).
  char buf[PAYLOAD_BUF_SIZE];
  size_t n = (length < (PAYLOAD_BUF_SIZE - 1)) ? length : (PAYLOAD_BUF_SIZE - 1);
  memcpy(buf, payload, n);
  buf[n] = '\0';

  trimInPlace(buf);
  if (buf[0] == '\0') return;   // ignore empty payloads

  handleCommand(buf);
}

/* ===========================================================================
 * WIFI / MQTT CONNECTIVITY
 * ======================================================================== */
static void connectWiFi() {
  Serial.printf("[WiFi] connecting to '%s' ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  // Retry loop with a bounded number of attempts before continuing;
  // loop() will keep WiFi alive thereafter (ESP8266 auto-reconnects).
  uint8_t attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 60) {
    delay(500);          // OK at boot; no MQTT traffic yet.
    Serial.print('.');
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] connected, IP=");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[WiFi] not connected yet; will keep trying in background.");
  }
}

/* Attempt a single (non-blocking-ish) MQTT reconnect. Returns true on success. */
static bool mqttReconnect() {
  if (WiFi.status() != WL_CONNECTED) return false;

  Serial.printf("[MQTT] connecting to %s:%d as '%s' ...\n",
                MQTT_HOST, MQTT_PORT, clientId);

  // Use STATUS_TOPIC as the Last-Will so subscribers know if we drop.
  bool ok = mqtt.connect(clientId,
                         NULL, NULL,                 // no auth
                         STATUS_TOPIC, 0, false,     // LWT topic, qos, retain
                         "{\"state\":\"offline\"}"); // LWT message

  if (ok) {
    Serial.println("[MQTT] connected.");
    mqtt.subscribe(COMMAND_TOPIC);
    Serial.printf("[MQTT] subscribed to '%s'\n", COMMAND_TOPIC);
    publishStatus("online");
  } else {
    Serial.printf("[MQTT] connect failed, rc=%d\n", mqtt.state());
  }
  return ok;
}

/* ===========================================================================
 * MOTION
 * ======================================================================== */

/* Advance the SCAN sweep target between SCAN_MIN and SCAN_MAX (millis-timed). */
static void updateScan(uint32_t now) {
  if (!scanMode) return;
  if (now - lastScanMs < SCAN_PERIOD) return;
  lastScanMs = now;

  targetAngle += scanDir * SCAN_STEP_DEG;

  if (targetAngle >= SCAN_MAX) {
    targetAngle = SCAN_MAX;
    scanDir = -1;
  } else if (targetAngle <= SCAN_MIN) {
    targetAngle = SCAN_MIN;
    scanDir = +1;
  }
  targetAngle = clampAngle(targetAngle);
}

/* Ease currentAngle toward targetAngle by at most EASE_STEP_DEG per tick. */
static void updateMotion(uint32_t now) {
  if (now - lastMotionMs < MOTION_PERIOD) return;
  lastMotionMs = now;

  float diff = targetAngle - currentAngle;
  if (diff > EASE_STEP_DEG)       currentAngle += EASE_STEP_DEG;
  else if (diff < -EASE_STEP_DEG) currentAngle -= EASE_STEP_DEG;
  else                            currentAngle = targetAngle;

  currentAngle = clampAngle(currentAngle);
  servo.write((int)lround(currentAngle));
}

/* ===========================================================================
 * ARDUINO SETUP / LOOP
 * ======================================================================== */
void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println();
  Serial.println("=== benax camera servo controller (ESP8266) ===");

  // Build a unique client id: base + chip id (hex).
  snprintf(clientId, sizeof(clientId), "%s-%06X",
           CLIENT_ID_BASE, ESP.getChipId());
  Serial.printf("[SYS] client id: %s\n", clientId);

  // Servo: attach and center.
  servo.attach(SERVO_PIN);
  servo.write((int)lround(currentAngle));
  Serial.printf("[SYS] servo on pin D4 (GPIO2), centered at %d deg\n",
                (int)lround(currentAngle));

  connectWiFi();

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
}

void loop() {
  uint32_t now = millis();

  // --- MQTT connection maintenance (non-blocking retry) ---
  if (!mqtt.connected()) {
    if (now - lastMqttTryMs >= MQTT_RETRY_PERIOD) {
      lastMqttTryMs = now;
      mqttReconnect();
    }
  } else {
    mqtt.loop();   // service incoming messages / keepalive
  }

  // --- Autonomous scan sweep ---
  updateScan(now);

  // --- Smooth servo easing ---
  updateMotion(now);

  // --- Periodic heartbeat ---
  if (now - lastHeartbeatMs >= HEARTBEAT_PERIOD) {
    lastHeartbeatMs = now;
    publishStatus("hb");
  }
}
