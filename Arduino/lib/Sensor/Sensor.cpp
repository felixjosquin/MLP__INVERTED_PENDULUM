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
}

void Sensor::init_X(Motor *motor)
{
    motor->write_motor(-LOW_SPEED);
    while (digitalRead(_pin_button) == LOW)
    {
    }
    motor->write_motor(0);
    _encodX->write(-MAX_X / 2);
    go_to(motor, 0);
}

void Sensor::go_to(Motor *motor, int value)
{
    if (abs(value) > MAX_X * 0.8 / 2)
    {
        return;
    }
    int delta = _encodX->read() - value, command;
    while (abs(delta) > 800)
    {
        command = -MAX_SPEED * ((delta > 0) - (delta < 0));
        motor->write_motor(command);
        delta = _encodX->read() - value;
    }
    motor->write_motor(0);
}

float Sensor::getTheta()
{
    int encodTheta_value = _encodTheta->read();
    while (encodTheta_value > ENCOD_THETA_PTS_HALF_REV || encodTheta_value <= -ENCOD_THETA_PTS_HALF_REV)
    {
        int offset = 2 * (1 - 2 * int(encodTheta_value > ENCOD_THETA_PTS_HALF_REV)) * ENCOD_THETA_PTS_HALF_REV;
        _encodTheta->write(encodTheta_value + offset);
        encodTheta_value += offset;
    }

    return encodTheta_value * 0.09; // value in deg
}

float Sensor::getX()
{
    int encodX_value = _encodX->read();
    return encodX_value * 0.0247202153279501; // value in mm
}

void Sensor::getDerivate() // CARFULLL TAKE 5 ms
{
    int beforeX = _encodX->read();
    int beforeTheta = _encodTheta->read();
    delay(5);
    _deltaX = _encodX->read() - beforeX;
    _deltaTheta = _encodTheta->read() - beforeTheta;
    _deltaTheta_available = true;
    _deltaX_available = true;
}

float Sensor::getdX(void (*error)())
{
    if (!_deltaX_available)
    {
        error();
    }
    _deltaX_available = false;
    return _deltaX * 0.0247202153279501 * 200;
}

float Sensor::getdTheta(void (*error)())
{
    if (!_deltaTheta_available)
    {
        error();
    }
    _deltaTheta_available = false;
    return _deltaTheta * 0.09 * 100.;
}
