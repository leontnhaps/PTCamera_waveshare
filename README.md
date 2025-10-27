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

2025-09-20
- 좀더 가시성 좋게하기 ✅
- yolo 학습 (dataset_4 이상)
ㄴ 우선 보정된 이미지 기반으로 학습 ✅
- 보내는거 문제가아니라 지금 찎는게 각도 바뀔때마다 안찎는거같은데 이제 이거 쓰레드 나눠야 되나싶기도하고 이부분 문제 다시생각하기 ✅ (이문제가생긴게 폴더 겹쳐서였는데 다른방식으로 찍어도 같은각도였어가지고 같은사진이 2번나온거였음)


할일
- 연구노트 밀린거 작성
- 각 필터별 인식 성능 생각해보기 (무슨 필터쓸지랑 빨강 고휘도 반사판에 어울리는 필터를 설계?)
- 카메라선 해결
- 데이터셋 학습 지표확인해보고 문제되는 부분 확인
- 이제 optical rail 관련 내일 공부하기 

2025-10-27
음 이제 리팩토링이랑 디렉토리 정리좀할거야

리팩토링 해야할거
1. 서버 ip 일일히 지정하지말고 서버에접속한 ip 확인해서 하기(가능하면 불가능하면 기존그대로 진행).
2. calib, yolo 같은거 최초에 코드적으로 지정하게 하고 추가 지정하게 바꾸기 ㅡㅡ 개귀찮음.
3. 쓸대없는 코드 싹다 밀어버리기 이제 디버깅 필요없으니까.
4. 코드를 두개짜야할거같아 실제 바로 할거랑 테스트용이랑 실제는 스캔 > 조정 바로되는거로 해야할거같은데


디렉토리 정리할거
1. 사용한 자료는 자료별로 묶어서 빼놓거나 하기.
2. 분석용 코드 따로 처리해놓기.

추가 연구 필요사항
1. 레이저 조준하는 알고리즘.
(지금생각중인건 1차 조준 후에 레이저 깜빡이는걸로 한번 인식해서 그 px 차이만큼 조준하는 방식?
생각중)
2. 여러개일때 객체 따로 인식하는 알고리즘.

Repo 정리(2025-10-27)

1. Com. Server. Raspverrypi. 는 그대로 냅두기

2. pointing~ 디렉들 ---> Fall_Paper_Datatset
내용 : 추계에 썼던 테스트 데이터셋들과 pointing 디버깅 데이터

3. angle_velocity --> Fall_Paper_Dataset
내용 : pan 과 px_w ,tilt와 px_h 와 선형인지 분석했던 코드와 데이터셋(angle_velocity_data)

4. Dataset --> Yolo 로 이름변경
내용 : 학습했던 Yolo 모델들 자료

5. Testset --> Calibration 으로 이름변경
내용 : chess data 자료
