import serial
import time

import matplotlib.pyplot as plt


# 포트 설정 (윈도우: 'COM3', 'COM4' 등 / 맥: '/dev/tty...' 확인 필요)
arduino_port = 'COM8' 
baud_rate = 9600 

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"아두이노({arduino_port})와 연결되었습니다.")
    time.sleep(2) # 아두이노가 리셋되고 안정화될 때까지 대기

    while True:
        if ser.in_waiting > 0:
            # 1. 데이터 한 줄 읽기
            line = ser.readline().decode('utf-8').strip()
            
            # 2. 쉼표(,)로 데이터 분리
            data = line.split(',')
            
            # 3. 데이터가 3개(전압,전류,전력) 제대로 들어왔는지 확인
            if len(data) == 3:
                voltage = data[0]
                current = data[1]
                power = data[2]
                
                print(f"☀️ 솔라셀 상태 -> 전압: {voltage}V | 전류: {current}mA | 전력: {power}mW")
                
                # 데이터를 그래프에 추가
                plt.plot(voltage, current, 'bo')
                plt.xlabel('Voltage (V)')
                plt.ylabel('Current (mA)')
                plt.title('Solar Panel Current vs Voltage')
                plt.grid(True)
                plt.pause(0.1)
            else:
                # 가끔 통신 찌꺼기 데이터가 들어오면 무시
                pass

except serial.SerialException:
    print(f"포트 {arduino_port}를 열 수 없습니다. 아두이노 IDE 시리얼 모니터가 켜져 있는지 확인하세요.")
except KeyboardInterrupt:
    print("프로그램을 종료합니다.")
    if 'ser' in locals() and ser.is_open:
        ser.close()