# Wiring, Power Architecture & Safety

## Connections

```
            ┌─────────────── ESP8266 (NodeMCU / Wemos D1 mini) ───────────────┐
 micro-USB  │  USB 5V ── on-board 3V3 regulator                               │
 (PC / hub) │                                                                  │
            │   D4 (GPIO2) ───────────────► Servo SIGNAL (orange/white)        │
            │   GND ──────────┬───────────► Servo GND  (brown/black)           │
            └─────────────────┼────────────────────────────────────────────────┘
                              │  COMMON GROUND (mandatory)
                              │
        External 5V supply ───┴── Servo V+ (red)
        (5V, ≥1A SG90 / ≥2A MG996R)
```

| Servo wire    | Connect to                              |
|---------------|------------------------------------------|
| Signal (orange/white) | ESP8266 **D4 / GPIO2**           |
| V+ (red)      | **External 5V** supply (preferred) or VIN |
| GND (brown)   | **Common ground** with ESP8266 GND        |

## Power architecture — why a separate servo supply

- The ESP8266's **3V3 pin cannot drive a servo** — stall/inrush current (0.5–2 A)
  browns out the 3V3 regulator and reboots the board. **Never power the servo from 3V3.**
- Best practice: a **dedicated 5V supply** for the servo, with its **ground tied to the
  ESP8266 ground** so the PWM signal has a common reference.
- A small SG90 *can* run off `VIN`/USB 5V for bench testing **if** the USB source supplies
  enough current (a powered USB hub, not a low-power laptop port). A geared MG996R should
  always use an external supply.
- Add a **470–1000 µF electrolytic capacitor** across the servo's V+/GND near the servo to
  absorb inrush spikes and reduce jitter/resets.

## Safety measures

- **Common ground** before applying power; a floating servo ground causes erratic motion.
- Respect the **0–180° clamp** in firmware; the 2-DOF mount's horizontal axis should have
  free travel across that range so the servo never stalls against a hard stop (stall =
  heat + current draw).
- Keep fingers/cables clear of the rotating camera arm; the SCAN sweep moves autonomously.
- Use the correct polarity on the external supply; reversed V+/GND can destroy the servo.
- Decouple: don't share the servo supply rail with the camera USB to avoid noise.
- Start with small `max_step_deg` (config) so motion is gentle during bring-up.

## First-power bring-up checklist

1. Flash firmware with your Wi-Fi SSID/PASS and `MQTT_HOST = 192.168.1.100`
   (the PC's Ethernet LAN IP; broker port 1884).
2. Power ESP via USB **first**, confirm Serial @115200 shows Wi-Fi + MQTT connect.
3. Apply external 5V to the servo (common ground already wired).
4. Publish a test command (see README) and confirm smooth motion.
5. If the camera moves the **wrong way**, set `tracking.invert_direction: true` in `config.json`.
