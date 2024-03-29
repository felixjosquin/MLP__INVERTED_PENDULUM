#include <Arduino.h>
#include <Encoder.h>

#include <Sensor.h>
#include <Motor.h>

Sensor::Sensor(int pin_encX_W, int pin_encX_G, int pin_encT_W, int pin_encT_O, int pin_button)
{
    _encodX = new Encoder(pin_encX_W, pin_encX_G);
    _encodTheta = new Encoder(pin_encT_W, pin_encT_O);
    _pin_button = pin_button;
    _deltaX_available = false;
    _deltaTheta_available = false;

    pinMode(_pin_button, INPUT);
    _time = new unsigned long();
}

void Sensor::initX(Motor *motor)
{
    _encodTheta->write(-ENCOD_THETA_PTS_REV / 2);
    motor->writeMotor(-LOW_SPEED);
    while (digitalRead(_pin_button) == LOW)
    {
    }
    motor->writeMotor(0);
    _encodX->write(-MAX_X / 2);
    goTo(motor, 0);
}

void Sensor::goTo(Motor *motor, int value)
{
    if (abs(value) > MAX_X * 0.8 / 2)
    {
        return;
    }
    int delta = _encodX->read() - value, command;
    while (abs(delta) > 800)
    {
        command = -LOW_SPEED * ((delta > 0) - (delta < 0));
        motor->writeMotor(command);
        delta = _encodX->read() - value;
    }
    motor->writeMotor(0);
}

float Sensor::getTheta()
{
    while (abs(_encodTheta->read()) > ENCOD_THETA_PTS_REV / 2)
    {
        int sign_theta = (_encodTheta->read() > 0) ? 1 : -1;
        _encodTheta->write(_encodTheta->read() - sign_theta * ENCOD_THETA_PTS_REV);
    }

    return _encodTheta->read() * 1.57079; // value in 1e-3 rad
}

float Sensor::getX()
{
    return _encodX->read() * 0.02472; // value in mm
}

bool Sensor::isTermined()
{
    return abs(_encodX->read()) > 6000 || abs(_encodTheta->read()) > 222; // -148mm<x<148mm and -19,98deg<theta<19,98 deg
}

bool Sensor::isTruncted()
{
    return false;
}

void Sensor::resetTime()
{
    *_time = micros();
}

unsigned long Sensor::getTime()
{
    return micros() - *_time;
}