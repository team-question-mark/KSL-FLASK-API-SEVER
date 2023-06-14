## PYTHON 3.7


## 참고 Git repo : https://github.com/kimjaebeom98/Sign-Language-Translator.git



## 설계 과정 

- AIhub 수어 영상 데이터 셋 및 자체 데이터 셋

![image](https://user-images.githubusercontent.com/87630540/193425726-253e7ba8-6d2c-42e5-a051-3686a44a62d4.png)

- MediaPipe의 Hollistic 솔루션을 이용하여 웹캠으로 부터 얻은 영상에서 왼손, 오른손 Keypoints를 추출

![image](https://user-images.githubusercontent.com/87630540/193425822-a4bd5ab2-3357-42c7-9d73-74391ad5ec68.png)

- 총 45개의 수어 단어 세트

![image](https://user-images.githubusercontent.com/87630540/193425878-4226f8d8-eb32-4126-9b6f-915ee9bfa097.png)

- LSTM 모델링은 tensorflow의 keras를 통해 진행되었다. input_shape=(timestep, feature)에서 timestep은 하나의 영상을 구성하는 프레임 개수, feature에는 126개의 왼 손, 오른 손 3D*(x, y, z) keypoints로 파라미터로 전달했다. 또 예측하고자 하는 target data인 수화 단어들(actions)을 출력층에 전달해 줬다. 또, 본 프로젝트 모델링에서는  hidden layer에 Stacked LSTM을 사용하여 LSTM이 더 복잡한 task를 해결할 수 있도록 LSTM 모델의 복잡도를 높혔다. 

![image](https://user-images.githubusercontent.com/87630540/193425920-52e3eaee-767e-48ec-aed1-a4915d9656c3.png)

- Flask-SocketIO를 이용하여 사용자로 부터 받은 영상을 실시간으로 처리하여 예측된 결과를 응답으로 보내줌

## 바뀐점
- 참고한 기존의git repo는 단어들을 조합하여 문장으로 쏘아주지만, 문장으로 바꾸지 않고 단어만을 쏘아주고, 다른 API서버를 통해서 문장으로 바꾸는 작업을 진행함
