#ifndef MOTOR_H

#define MOTOR_H

#define LOW_SPEED 100
#define MAX_SPEED 255

class Motor
{
public:
    Motor(int pin_ENB, int pin_1, int pin_2);
    void writeMotor(int state);
    int readMotor();

private:
    int _state;
    int _pin_ENB;
    int _pin_1;
    int _pin_2;
};

#endif