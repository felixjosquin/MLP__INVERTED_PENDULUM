#ifndef SERIAL_USB_H
#define SERIAL_USB_H
class Sensor;

void wait_start();
void send_data(Sensor *sensor);

#endif