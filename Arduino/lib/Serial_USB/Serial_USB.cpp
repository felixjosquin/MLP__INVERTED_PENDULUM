#include <Arduino.h>
#include <ArduinoJson.h>

#include <Serial_USB.h>
#include <Sensor.h>

USBCom::USBCom()
{
    _command = 0;
}

void USBCom::waitBeginning()
{
    String receivedString = "";
    while (true)
    {
        if (Serial.available())
        {
            receivedString = Serial.readStringUntil('\n');
            if (receivedString == "start")
            {
                Serial.println("start");
                return;
            }
        }
    }
}

// status 0 = No need new command
// status 1 = Need new command
// status 2 = End episode
// status 3 = error
void USBCom::sendData(Sensor *sensor, int status)
{

    JsonDocument jsonData;

    jsonData["s"] = status;
    jsonData["t"] = sensor->getTime();
    jsonData["x"] = sensor->getX();
    jsonData["th"] = sensor->getTheta();
    jsonData["c"] = _command;

    serializeJson(jsonData, Serial);
    Serial.print('\n');
    Serial.flush();
};

int USBCom::receiveCommand(void (*error)())
{
    String jsonString = Serial.readStringUntil('\n');
    if (!jsonString.length())
    {
        error();
    }
    JsonDocument doc;
    deserializeJson(doc, jsonString);
    _command = doc["c"];
    return _command;
}

void USBCom::endEpisode()
{
    _command = 0;
}
