#ifndef SENSOR_H

#define SENSOR_H

#define MAX_X 14000
#define ENCOD_THETA_PTS_REV 4000

class Motor;
class Encoder;

class Sensor
{
public:
    Sensor(int pin_encX_W, int pin_encX_G, int pin_encT_W, int pin_encT_O, int pin_button);
    void initX(Motor *motor);
    void goTo(Motor *motor, int value);
    float getTheta();
    float getX();
    bool isTermined();
    bool isTruncted();
    void resetTime();
    unsigned long getTime();

private:
    int _pin_button;
    Encoder *_encodX;
    Encoder *_encodTheta;
    unsigned long _deltaX;
    unsigned long _deltaTheta;
    bool _deltaTheta_available;
    bool _deltaX_available;
    unsigned long *_time;
};
#endif