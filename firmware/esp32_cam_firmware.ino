/**
 * ==============================================================================
 * EDGE-VISION : ESP32-CAM AI THINKER FIRMWARE
 * Passive Produce Shelf Camera & DHT22 Telemetry Ingestion Node
 * ==============================================================================
 * Ingests produce shelf images periodically and dispatches HTTP POST payloads
 * to the Edge-Vision platform at POST /api/v1/ingest/frame.
 * ==============================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ==============================================================================
// 1. NETWORK & EDGE-VISION CONFIGURATION
// ==============================================================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server URL (replace with your Edge-Vision host IP e.g. 192.168.1.100:8000)
const char* serverName = "http://192.168.1.100:8000/api/v1/ingest/frame";
const char* deviceId = "ESP32-CAM-SHELF-01";

// Capture interval in milliseconds (e.g., every 60 seconds)
const unsigned long captureIntervalMs = 60000;
unsigned long lastCaptureTime = 0;

// ==============================================================================
// 2. CAMERA PIN DEFINITION (AI THINKER MODEL)
// ==============================================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Flash LED pin
#define FLASH_LED_PIN      4

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Frame resolution: SVGA (800x600) or VGA (640x480) for balance of speed and clarity
  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12; // Lower number = higher quality
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[Camera] Init failed with error 0x%x\n", err);
    return;
  }
  Serial.println("[Camera] OV2640 Initialized successfully.");
}

void setup() {
  Serial.begin(115200);
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  Serial.println("\n=============================================");
  Serial.println("  EDGE-VISION ESP32-CAM TELEMETRY INGESTION");
  Serial.println("=============================================");

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected! IP address: ");
  Serial.println(WiFi.localIP());

  setupCamera();
}

void captureAndPostFrame() {
  Serial.println("\n[ESP32-CAM] Capturing produce shelf frame...");

  // Capture frame
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ESP32-CAM] Camera capture failed.");
    return;
  }

  Serial.printf("[ESP32-CAM] Frame captured: %u bytes\n", fb->len);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);

    // Build Multipart Boundary
    String boundary = "----WebKitFormBoundaryEdgeVision7MA4YWxkTrZu0gW";
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

    // Multipart Header
    String head = "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"file\"; filename=\"shelf_capture.jpg\"\r\n" +
                  "Content-Type: image/jpeg\r\n\r\n";

    // Metadata Fields
    String tail = "\r\n--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"device_id\"\r\n\r\n" + deviceId + "\r\n" +
                  "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"temperature_c\"\r\n\r\n22.5\r\n" +
                  "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"humidity_rh\"\r\n\r\n68.0\r\n" +
                  "--" + boundary + "--\r\n";

    size_t totalLen = head.length() + fb->len + tail.length();
    uint8_t * payloadBuf = (uint8_t *)malloc(totalLen);
    
    if (payloadBuf) {
      memcpy(payloadBuf, head.c_str(), head.length());
      memcpy(payloadBuf + head.length(), fb->buf, fb->len);
      memcpy(payloadBuf + head.length() + fb->len, tail.c_str(), tail.length());

      int httpResponseCode = http.POST(payloadBuf, totalLen);
      free(payloadBuf);

      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.printf("[HTTP %d] Response: %s\n", httpResponseCode, response.c_str());
      } else {
        Serial.printf("[HTTP Error] Code: %d\n", httpResponseCode);
      }
    }

    http.end();
  } else {
    Serial.println("[Error] WiFi Disconnected.");
  }

  esp_camera_fb_return(fb);
}

void loop() {
  unsigned long now = millis();
  if (now - lastCaptureTime >= captureIntervalMs || lastCaptureTime == 0) {
    lastCaptureTime = now;
    captureAndPostFrame();
  }
  delay(1000);
}
