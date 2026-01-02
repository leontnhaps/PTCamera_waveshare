// [OWPT 수신단 - 3비트(000~111) 순차 점등 테스트]
// 5m 거리 시인성 테스트용

const int ledPins[] = {3, 4, 5}; // 3번이 1의 자리, 5번이 4의 자리

void setup() {
  for (int i = 0; i < 3; i++) {
    pinMode(ledPins[i], OUTPUT);
  }
}

void loop() {
  // 0부터 7까지 숫자(count)를 1씩 증가시킴
  for (int count = 0; count < 8; count++) {
    
    // 각 숫자를 2진수로 변환해서 LED 켜기
    // 예: count가 3이면 (2진수 011) -> 핀3 ON, 핀4 ON, 핀5 OFF
    for (int i = 0; i < 3; i++) {
      if (bitRead(count, i) == 1) {
        digitalWrite(ledPins[i], HIGH);
      } else {
        digitalWrite(ledPins[i], LOW);
      }
    }

    // 2초 대기 (사진 찍을 시간 확보)
    delay(2000); 
  }
}