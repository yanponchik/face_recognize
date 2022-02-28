import cv2
import os
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
cv2.startWindowThread()
cap = cv2.VideoCapture(0)
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt


# extract a single face from a given photograph
def extract_face(faceimg, required_size=(224, 224)): #функция выделяет квадрат лица из фото
    global g   #переменная нужна для того, чтобы проверять есть ли лицо в кадре
    print(type(faceimg))
        #if type(faceimg) == str:
            #pixels = pyplot.imread(faceimg)
        #else:
    pixels = faceimg
    detector = MTCNN() #MTCNN распознает лицо

    results = detector.detect_faces(pixels)
    if results != []:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array   #возвращает квадрат лица
    else:
        face = pixels[0:1, 0:1]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        g = 1
        return face_array #возвращает несколько пикселей, нужно лишь для того, чтобы отсеивать кадр без лиц

def get_model_scores(faces):
    global g
    if faces != []:

        samples = asarray(faces, 'float32')

        # prepare the data for the model
        samples = preprocess_input(samples, version=2)

        # create a vggface model object
        model = VGGFace(model='resnet50',
          include_top=False,
          input_shape=(224, 224, 3),
          pooling='avg')

        # perform prediction
        return model.predict(samples) # возвращает черты лица в векторной форме

while True:
    g = 0
    img, frame = cap.read()  #переменная frame получает кадр с камеры
    #out.write(frame.astype('uint8'))
    #pixels = extract_face(frame)
    directory = 'FACE0'  #дериктория в которой хранятся лица
    files = os.listdir(directory)  #список названия файлом лиц
    print(files)
    for i in files:
        l = pyplot.imread('./FACE0/' + i)  #из названия получает саму фотку
        faces = [extract_face(faceimg)
                 for faceimg in [frame, l]]  #отправляет кадр и фото из папки в функцию extract_fa
        #print(faces)
        #print(faces[0])
        #if None not in faces:
        #help = faces[0]
        #print(help)
        #help2 = faces[1]
        #print(help2)
        if g != 1:
            model_scores = get_model_scores(faces) #отправляет список фото лиц, которые уже прошли через функцию extract_face
            if cosine(model_scores[0], model_scores[1]) <= 0.4: #Функция вычисляет расстояние между двумя косинуса векторов. Чем меньше чесло тем лучше он распознает лица
                print("Faces Matched") #вывод лица совпали
            else:
                print('Faces not Matched') #вывод лица не совпали
        else:
            g = 0
            print('no face') #вывод лицо не найдено
