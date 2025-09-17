# PTCamera_waveshare
control for https://www.waveshare.com/2-axis-pan-tilt-camera-module.htm 

2025-09-15

1. ip 확인 > 라파 코드에 주석으로 처리 ✅
2. GPU 사용가능하면 하자 .. (속도향상 이건 open cv 를 torch 에 맞게 콘다 환경을 바꿔야할듯) ✅
2.1 GUI 기능체크 ✅
- Scan ✅
- Manual / LED
ㄴ Pan,Tilt Led ✅
- Preview & Settings
ㄴ Live Preview : 프리뷰 끄고키기 , 설정값 넘어감 ✅
ㄴ Preview 설정값들 : w/h : 2592,1944 1920,1080 640,360 , fps : frame per seconds  , quality : 품질 낮춰서 preview 하기 ✅
ㄴ Apply Preview Size : 설정값들 preview 에 적용하기 ✅
ㄴ Undistort preview , Load calib.npz : 보정값 불러오고 프리뷰 적용 ✅
ㄴ Also save undistorted copy : 보정한버젼도 저장 ✅
ㄴ Alpha/Balance (0~1) : 보정 계수 ✅
3. Scan data set 
ㄴ pan tilt step : 10 , Speed 100, Accel 1.0 Settle 0.5 -> 흔들림 제어 ✅
ㄴ 이상태에서 pan ,tilt step 30 변경시 흔들림 제어 ❌
ㄴㄴ speed 는 줄여도 체감안됨
ㄴㄴ Accel 는 줄이면 체감이됨
ㄴ pan tilt step : 30 , Speed 100, Accel 0.5 Settle 0.6 -> 흔들림 제어 ✅


2025-09-17
1. Dataset_3 까지만 우선 학습시켜서 코드에 적용함
ㄴ 성능은 생각보다 Not bad 하지만 중간중간 인식 안되는거 생각하면 좀더 학습을하던가 파라미터 조정을 하던가 방안을 생각해야할듯


** 반사판은 가로줄 기준으로 부착?


문제 1. 거리가 멀면 반사가 잘안됨.. (problem 폴더)
문제 2. 스캔각을 잘해야 반사가 될듯?



할일
- yolo 학습 (dataset_4 이상)
ㄴ 우선 보정된 이미지 기반으로 학습
- 연구노트 밀린거 작성
- 각 필터별 인식 성능 생각해보기 (무슨 필터쓸지랑 빨강 고휘도 반사판에 어울리는 필터를 설계?)
- 카메라선 해결

- 좀더 가시성 좋게하기
- 
