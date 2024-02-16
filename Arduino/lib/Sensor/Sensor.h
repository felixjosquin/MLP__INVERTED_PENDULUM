#ifndef SENSOR_H

#define SENSOR_H

#define MAX_X 14000
#define ENCOD_THETA_PTS_HALF_REV 2000

class Motor;
class Encoder;

class Sensor
{
public:
    Sensor(int pin_encX_W, int pin_encX_G, int pin_encT_W, int pin_encT_O, int pin_button);
    void init_X(Motor *motor);
    void go_to(Motor *motor, int value);
    float getTheta();
    float getX();
    float getdX(void (*func)());
    float getdTheta(void (*func)());
    void getDerivate();

private:
    int _pin_button;
    Encoder *_encodX;
    Encoder *_encodTheta;
    int _deltaX;
    int _deltaTheta;
    bool _deltaTheta_available;
    bool _deltaX_available;
};
#endif