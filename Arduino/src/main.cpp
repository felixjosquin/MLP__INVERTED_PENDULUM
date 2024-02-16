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

void error()
{
  motor->write_motor(0);
  Serial.println("{\"status\":\"error\"}");
  delay(100);
  exit(0);
}

void setup()
{
  Serial.begin(9600);
  wait_start();
  sensor->init_X(motor);
}

void loop()
{
  sensor->go_to(motor, -5000);
  sensor->go_to(motor, 5000);
}
