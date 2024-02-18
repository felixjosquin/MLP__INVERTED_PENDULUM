#ifndef SERIAL_USB_H
#define SERIAL_USB_H
class Sensor;

class USBCom
{
public:
    USBCom();
    void sendData(Sensor *sensor, int status);
    int receiveCommand(void (*error)());
    void waitBeginning();
    void endEpisode();

private:
    int _command;
};

#endif