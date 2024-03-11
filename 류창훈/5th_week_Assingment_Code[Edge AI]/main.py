from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
import cv2
import numpy as np
import tensorflow as tf


Builder.load_string('''
<MyLayout>:
    orientation: 'vertical'
''')

class AndroidCamera(Widget):
    def __init__(self, **kwargs):
        super(AndroidCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.interpreter = tf.lite.Interpreter(model_path="detection_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def start_detection(self):
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return
        
       
        input_image = cv2.resize(frame, (28, 28))
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0
        
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        label = np.argmax(preds)
        confidence = preds[label]

        if confidence > 0.6:  # 임계값 설정(0.6 이상이면 bounding box)
            self.draw_rectangle(label)

    def draw_rectangle(self, label):
        with self.canvas:
            if label == 0:  # apple red box
                Color(1, 0, 0, 0.5)  
            elif label == 1:  # banana yellow box
                Color(1, 1, 0, 0.5)  
            Rectangle(pos=self.pos, size=self.size)

class MyLayout(BoxLayout):
    pass

class ObjectDetectionApp(App):
    def build(self):
        layout = MyLayout()
        self.camera = AndroidCamera()
        layout.add_widget(self.camera)
        btn = Button(text='Start Detection')
        btn.bind(on_press=lambda instance: self.camera.start_detection())
        layout.add_widget(btn)
        return layout

if __name__ == '__main__':
    ObjectDetectionApp().run()
