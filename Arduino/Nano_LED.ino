// [RX 시스템 전압 감시 및 LED 표시 코드]
const int redPin = 3;    // Red LED
const int greenPin = 4;  // Green LED
const int bluePin = 5;   // Blue LED
const int batteryPin = A0;

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  Serial.begin(9600); // PC에서 전압 확인용
}

void loop() {
  // 1. 배터리 전압 읽기 (10번 측정해서 평균냄 - 노이즈 제거)
  long sum = 0;
  for(int i=0; i<10; i++) {
    sum += analogRead(batteryPin);
    delay(10);
  }
  float avgRead = sum / 10.0;
  
  // 2. 실제 전압 계산 (5.0V 승압 기준)
  float voltage = (avgRead * 5.0) / 1023.0;
  
  // 시리얼 모니터로 전압 확인 (필요시 보정용)
  Serial.print("Battery: "); Serial.print(voltage); Serial.println("V");

  // 3. 배터리 잔량 구간별 LED 제어
  allOff();
  if (voltage < 3.7) {
    digitalWrite(redPin, HIGH);   // 0~40% : 빨강 (충전 시급)
  } 
  else if (voltage < 4.0) {
    digitalWrite(greenPin, HIGH); // 40~80% : 초록 (보통)
  } 
  else {
    digitalWrite(bluePin, HIGH);  // 80~100% : 파랑 (충전 완료)
  }

  delay(500); // 0.5초마다 갱신
}

void allOff() {
  digitalWrite(redPin, LOW);
  digitalWrite(greenPin, LOW);
  digitalWrite(bluePin, LOW);
}