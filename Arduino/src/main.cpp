#include <Encoder.h>
#include <Arduino.h>

#include <Motor.h>
#include <Sensor.h>
#include <Serial_USB.h>

#define PIN_ENCX_W 2
#define PIN_ENCX_G 5

#define PIN_ENCT_W 3
#define PIN_ENCT_B 4

#define PIN_MOTOR_1 7
#define PIN_MOTOR_2 8
#define PIN_MOTOR_ENB 9

#define PIN_BUTTON 12

Motor *motor = new Motor(PIN_MOTOR_ENB, PIN_MOTOR_1, PIN_MOTOR_2);
Sensor *sensor = new Sensor(PIN_ENCX_W, PIN_ENCX_G, PIN_ENCT_W, PIN_ENCT_B, PIN_BUTTON);
USBCom *usbcom = new USBCom();

void error()
{
  motor->writeMotor(0);
  Serial.println("{\"s\":3}");
  delay(100);
  exit(0);
}

void setup()
{
  Serial.begin(115200);
  Serial.setTimeout(20);
  usbcom->waitBeginning();
  sensor->initX(motor);
}

void loop()
{
  while (abs(sensor->getTheta()) > 34.9)
  {
    delay(1);
  }
  int status = 1;
  sensor->resetTime();
  usbcom->sendData(sensor, status);
  int command = usbcom->receiveCommand(&error);
  motor->writeMotor(command);
  while (status != 2)
  {
    if (sensor->isTermined() || sensor->isTruncted())
    {
      status = 2; // end of episode
    }
    else
    {
      status = 1;
    }
    usbcom->sendData(sensor, status);
    if (status == 1)
    {
      int command = usbcom->receiveCommand(&error);
      motor->writeMotor(command);
    }
  }
  motor->writeMotor(0);
  usbcom->endEpisode();
  sensor->goTo(motor, 0);
}

// void loop()
// {
//   Serial.println(sensor->getTheta());
//   delay(100);
// }