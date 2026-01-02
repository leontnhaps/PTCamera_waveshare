#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

void setup() {
  Serial.begin(9600); // 통신 속도
  
  // INA219 칩 시작
  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip");
    while (1) { delay(10); }
  }
}

void loop() {
  float busVoltage_V = 0;
  float current_mA = 0;
  float power_mW = 0;

  // 값 읽기
  busVoltage_V = ina219.getBusVoltage_V();
  current_mA = ina219.getCurrent_mA();
  power_mW = ina219.getPower_mW();

  // 데이터 전송 (전압,전류,전력)
  Serial.print(busVoltage_V);
  Serial.print(",");
  Serial.print(current_mA);
  Serial.print(",");
  Serial.println(power_mW);

  delay(500); 
}