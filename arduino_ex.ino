#include <Servo.h>
Servo myservo;
void setup() {
  myservo.attach(10);//attach servo motor PWM(orange) wire to pin 10 
  pinMode(0, INPUT);//attach GPIO 7&8 pins to arduino pin 0&1
  pinMode(1,INPUT);
}
void loop() {
           if(digitalRead(0)==HIGH && digitalRead(1)==LOW)
                {
                      myservo.write(118);
                }
          if(digitalRead(1)==HIGH && digitalRead(0)==LOW)
                {
                      myservo.write(62);
                }
          if(digitalRead(1)==LOW && digitalRead(0)==LOW)
                {
                       myservo.write(90);
                } 
}
