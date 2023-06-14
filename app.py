from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import time, io, os, time, sys, natsort, random, math
from PIL import Image
import base64,cv2
import numpy as np
import joblib
# import library 
import pandas as pd
import mediapipe as mp
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
# SocketIO는 ‘app’에 적용되고 있으며 나중에 애플리케이션을 실행할 때 앱 대신 socketio를 사용할 수 있도록 socketio 변수에 저장된다.
socketio = SocketIO(app,cors_allowed_origins='*' )


# @socketio.on('connectKSL')
# def on_connect():
#     emit('connectKSL', 'KSL SERVER CONNECTED!')  
    

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# 왼손, 오른손 key_point 추출
def extract_keypoints(results):
    lh = np.array([[res.x*3, res.y*3, res.z*3] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x*3, res.y*3, res.z*3] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)    
    return np.concatenate([lh, rh])

my_dict ={"None":0, "계산":1, "고맙다":2, "괜찮다":3, "기다리다":4, "나" :5, "네": 6,
          "다음":7, "달다":8, "더":9, "도착":10, "돈":11, "또":12, 
          "맵다":13, "먼저":14, "무엇":15, "물":16, "물음":17, "부탁":18, "사람":19, 
          "수저":20, "시간":21, "아니요":22, "어디":23, "얼마":24,"예약":25,
          "오다":26, "우리":27, "음식":28, "이거":29, "인기":30, "있다":31, "자리":32,
          "접시":33, "제일":34, "조금":35, "주문":36, "주세요":37, "짜다":38, "책":39,
          "추천":40, "화장실":41, "확인":42}

## 받은 5개의 단어들을 데이터 프레임으로 변환 
def make_word_df(word0, word1, word2, word3, word4):
    info = [[word0, word1, word2, word3, word4]]
    df = pd.DataFrame(info, columns = ['target0', 'target1', 'target2', 'target3', 'target4'])
    return df

## 받은 단어를 숫자로 반환
def get_key(val):
    for key, value in my_dict.items():
         if val == key:
             return value
 
    return "There is no such Key"

## 인자로 받은 단어 5개의 데이터프레임을 
def make_num_df(input_1):
    num_oflist = []
    for i in input_1.columns:
        num_oflist.append(get_key(input_1[i].values))
    input2 = make_word_df(num_oflist[0], num_oflist[1], num_oflist[2], num_oflist[3], num_oflist[4])
    return input2

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

# Actions that we try to detect
actions = np.array(['None', '계산', '고맙다', '괜찮다', '기다리다', '나', '네', '다음',
                   '달다', '더', '도착', '돈', '또', '맵다', '먼저', '무엇', '물', '물음',
                   '부탁', '사람', '수저', '시간', '아니요', '어디', '얼마', '예약', '오다',
                   '우리', '음식', '이거', '인기', '있다', '자리', '접시', '제일', '조금',
                   '주문', '주세요', '짜다', '책', '추천', '화장실', '확인'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss ='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights("./actionxhand_data0524_0513.h5")
rlf = joblib.load("./sentence_model.pkl")
data = pd.read_excel("./sentence_data.xlsx", engine = 'openpyxl')
data_x = data.drop(['sentence'], axis = 1)
data_y = data['sentence']
le = LabelEncoder()
le.fit(data['sentence'])


blue_color = (255,0,0)




def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    # Take in base64 string and return PIL image
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    # convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)

# 서버가 클라이언트가 보낸 메시지를 받는 방법, 클라이언트의 메시지를 확인하는 방법
# catch-frame 이벤트 핸들러 정의
# catch-frame 를 트리거 할 때 response_back 이벤트로 전송함 2번째 인자 data와 같이
@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  

global count, sequence, sentece, predictions
sequence = []
sentence = []
predictions = []
result_array = [] #result로 보낼 배열
count = 0

@app.get('/ho')
def abc():
    print('dijiejfiej'    )
    return 'as sdfe'


# 여기는 단어만을 계속해서 보낼때 사용하는 로직

#                 #예측한 단어의 정확도가 0.5를 넘는다면
#                 if res[np.argmax(res)] > threshold: 
#                     latest_word = actions[np.argmax(res)]
#                     if len(sentence) > 0: 
#                         if latest_word == 'None':
#                             print("None 찍힘 : " + latest_word)
#                             sentence.append(latest_word)
#                         else:
#                             print("정상적으로 들어옴 : " + latest_word)
#                             if(latest_word != sentence[-1]):
#                                 sentence.append(latest_word)
#                     else:
#                         sentence.append(latest_word)
#                     print("결과 : " + latest_word)
#                     emit('result', latest_word)
                    
#                 count = 0
#                 sequence.clear()

#             emit('start', 'start')





# image 이벤트 핸들러 정의 클라이언트에서 image 이벤트 보냄
@socketio.on('image')
def image(data_image):
    #print("이미지 socket 실행완료") 
    global sequence, sentence, predictions, count, result_array
    threshold = 0.5


    # 사실상 delete는 사용하지 않지만 넣어놓음
    if(data_image == "delete"):
        if(len(sentence) != 0):
            sequence = [] 
            count = 0
            delete_word = sentence[-1]
            sentence.pop(-1)
            delete_word = delete_word + "가 삭제되었습니다."
            emit('delete_back', delete_word)
        else:
            delete_word = "번역된 단어가 없습니다."
            emit('delete_back', delete_word)
        return
    else:
        frame = (readb64(data_image))
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            count = count+1

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            print(" sequence len : "+str(len(sequence)))
            if (len(sequence) % 30 == 0):
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #예측한 단어의 정확도가 0.5를 넘는다면
                if res[np.argmax(res)] > threshold: 
                    latest_word = actions[np.argmax(res)]
                    print(" 예측된 단어 : "+latest_word)
                    if latest_word == 'None':
                        print(" 보낼 result : "+ str(result_array))
                        #결과 보내기
                        emit('result',result_array)
                        result_array = []  #배열 초기화
                    else:
                        # 단어 중복방지 -> 마지막 단어와 새로 추가된 단어가 다를때만 배열에 넣기
                        if len(result_array) == 0 or (len(result_array) > 0 and result_array[-1] != latest_word):
                            result_array.append(latest_word)
                            print(" 담긴 result_array : " + str(result_array))

                    
                count = 0
                sequence.clear()

            emit('start', 'start')            
       

if __name__ == '__main__':
    #socketio.run(app ,debug=True)
    app.run(host='0.0.0.0')
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
