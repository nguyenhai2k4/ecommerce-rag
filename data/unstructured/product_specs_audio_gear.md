# Product Specifications & Troubleshooting Guide: Audio & Smart Gear

**Document ID:** SPEC-TECH-2026-A1  
**Category:** Audio, Wearables & Charging Accessories

---

## 1. AeroSound Pro Wireless ANC Headphones (SKU: SKU-AUD-001)

### 1.1 Technical Specifications
- **Form Factor:** Over-Ear Wireless Headphones
- **Driver Size:** 40mm Custom Bio-Cellulose Dynamic Drivers
- **Active Noise Cancellation (ANC):** Hybrid ANC with 4 beamforming microphones (-38dB reduction)
- **Bluetooth Version:** Bluetooth 5.3 (Multipoint connection up to 2 devices)
- **Supported Audio Codecs:** LDAC, AAC, SBC
- **Battery Capacity:** 800mAh Lithium-Polymer
  - Playback Time (ANC On): 35 Hours
  - Playback Time (ANC Off): 50 Hours
  - Quick Charge: 10 minutes provides 5 hours playback
- **Water Resistance Rating:** IPX4 (Splash & sweat resistant; not submersible)
- **Weight:** 255g

### 1.2 Common Troubleshooting Procedures
- **Bluetooth Pairing Reset:**
  1. Power off the headset.
  2. Press and hold the **Power Button** and **Volume Down (-)** simultaneously for 8 seconds until the LED flashes rapid magenta.
  3. Delete "AeroSound Pro" from your device's Bluetooth settings and re-pair.
- **Audio Distortion or Crackling:**
  - Verify codec compatibility in device settings. When using LDAC on crowded 2.4GHz Wi-Fi channels, switch Bluetooth Quality to "Best Effort / Adaptive Bitrate".

---

## 2. PulseFit Smart Tracker Band (SKU: SKU-SMT-004)

### 2.1 Technical Specifications
- **Display:** 1.47-inch AMOLED, 368x194 resolution, Always-On Display supported
- **Sensors:** Photoplethysmography (PPG) Heart Rate, SpO2 Blood Oxygen, 6-Axis Accelerometer + Gyroscope
- **Battery Life:** Up to 14 days standard usage; 7 days heavy usage with Continuous SpO2 enabled
- **Water Rating:** 5 ATM (Water-resistant up to 50 meters, suitable for swimming)
- **Companion App:** NexusFit OS (iOS 15+, Android 10+)

### 2.2 Known Firmware Advisories
- Firmware v1.2 experienced a sleep phase tracking offset where light sleep was miscategorized as awake time. Firmware v1.2.4 released in August 2026 fully resolves this anomaly.

---

## 3. UltraCharge 65W GaN Fast Charger (SKU: SKU-PWR-007)

### 3.1 Technical Specifications
- **Architecture:** Gallium Nitride (GaN III) Semiconductor
- **Ports:** 2x USB-C (Power Delivery 3.0 / PPS), 1x USB-A (Quick Charge 4+)
- **Power Distribution:**
  - Single Port C1: 65W Max
  - Dual Port C1 + C2: 45W + 20W
  - Triple Port C1 + C2 + A: 45W + 10W + 10W
- **Thermal Safety:** Intelligent Over-Temperature Protection triggers automated throttle if internal temperature exceeds 75°C.
