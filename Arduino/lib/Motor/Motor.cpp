#include <Arduino.h>

#include <Motor.h>

Motor::Motor(int pin_ENB, int pin_1, int pin_2)
{
    pinMode(pin_1, OUTPUT);
    pinMode(pin_2, OUTPUT);
    pinMode(pin_ENB, OUTPUT);

    _state = 0;
    _pin_ENB = pin_ENB;
    _pin_1 = pin_1;
    _pin_2 = pin_2;
}

void Motor::write_motor(int new_state)
{
    if (new_state >= 0)
    {
        digitalWrite(_pin_1, HIGH);
        digitalWrite(_pin_2, LOW);
    }
    else
    {
        digitalWrite(_pin_1, LOW);
        digitalWrite(_pin_2, HIGH);
    }
    analogWrite(_pin_ENB, abs(new_state));
    _state = new_state;
}

int Motor::read_motor()
{
    return _state;
}
