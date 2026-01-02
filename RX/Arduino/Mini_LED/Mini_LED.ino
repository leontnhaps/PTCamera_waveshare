
// [OWPT 수신단 - 프로 미니용 최종 코드]

// 1. 자신의 프로 미니 버전에 맞게 수정하세요!
float VCC_VOLTAGE = 5.0; // 5V 모델이면 5.0, 3.3V 모델이면 3.3으로 수정

// 2. 핀 설정 (설계도 기준)
const int redPin = 3;    // D3: Red LED
const int greenPin = 4;  // D4: Green LED
const int bluePin = 5;   // D5: Blue LED
const int batteryPin = A0; // A0: 배터리 전압 감지

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  
  // 시리얼 모니터로 전압 확인 (FTDI 어댑터 연결 상태에서 확인 가능)
  Serial.begin(9600); 
}

void loop() {
  // 3. 배터리 전압 읽기 (10번 측정 평균 필터 - YOLO 인식 안정화용)
  long sum = 0;
  for(int i=0; i<10; i++) {
    sum += analogRead(batteryPin);
    delay(10);
  }
  float avgRead = sum / 10.0;
  
  // 4. 전압 계산 공식 (프로 미니 VCC 기준)
  // 측정 전압 = (아날로그 읽기값 * 보드 작동 전압) / 1023
  float voltage = (avgRead * VCC_VOLTAGE) / 1023.0;
  
  // PC 시리얼 모니터 출력
  Serial.print("Current Battery Voltage: ");
  Serial.print(voltage);
  Serial.println(" V");

  // 5. 배터리 잔량 구간별 LED 제어
  allLedsOff();
  
  if (voltage < 3.7) {           // 약 0~40% 구간
    digitalWrite(redPin, HIGH);   // 빨간색 점등
  } 
  else if (voltage < 4.0) {      // 약 40~80% 구간
    digitalWrite(greenPin, HIGH); // 초록색 점등
  } 
  else {                         // 약 80~100% 구간
    digitalWrite(bluePin, HIGH);  // 파란색 점등
  }

  delay(500); // 0.5초마다 상태 갱신
}

void allLedsOff() {
  digitalWrite(redPin, LOW);
  digitalWrite(greenPin, LOW);
  digitalWrite(bluePin, LOW);
}


// 아두이노 프로 미니의 내장 LED는 보통 13번 핀에 연결되어 있습니다.
/*
void setup() {
  // 13번 핀을 전기를 보내는 '출력' 모드로 설정합니다.
  pinMode(13, OUTPUT);
}

void loop() {
  digitalWrite(13, HIGH);   // LED를 켭니다 (전압 UP)
  delay(1000);              // 1초(1000ms) 동안 기다립니다.
  digitalWrite(13, LOW);    // LED를 끕니다 (전압 DOWN)
  delay(1000);              // 1초 동안 기다립니다.
}
*/