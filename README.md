Идея:
	Написать ПО для контроля посещаемости уроков учащимися, используя
	библиотеку Opencv и обученную нейросеть TensorFlow.


Основные функции
	* Отчёт о посещаемости для кл.рук. с отправкой фотографий в Telegram.
	* Система распознавания ,,Свой - чужой,, (если на уроке находится ученик из другого класса)
	* Получение списка Присутствующих/отсутствующих на уроке.


ТЗ для MVP (минимально жизнеспособный продукт)
Аппаратно:
	Raspberry Pi 4 + веб-камера


Программно:
	Поиск лиц(MTCNN Library)  + сравнение лица с созданной заранее б/д (VGG).
	Использование API Telegram для отправки сообщений боту.
	Вывод списка учащихся присутствующих на уроке. 
