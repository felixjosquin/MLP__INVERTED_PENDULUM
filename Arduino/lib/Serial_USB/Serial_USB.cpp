#include <Arduino.h>
#include <ArduinoJson.h>

#include <Serial_USB.h>
#include <Sensor.h>

void wait_start()
{
    String receivedString = "";
    bool wait = true;
    while (wait)
    {
        if (Serial.available())
        {
            char incomingChar = Serial.read();
            receivedString += incomingChar;
            if (receivedString.endsWith("start"))
            {
                wait = false;
            }
        }
    }
}

void send_data(Sensor *sensor){

};
