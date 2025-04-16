#include "mbed.h"

Serial pc(USBTX, USBRX);  // ST-Link Virtual COM port

const int buffer_size = 64;
char buffer[buffer_size];
int index = 0;
bool manualMode = false;

int main() {
    pc.baud(9600);
    pc.printf("Ready to receive coordinates...\r\n");
    pc.printf("Press 'm' for manual coordinate entry mode.\r\n");

    while (true) {

        if (pc.readable()) {
            char c = pc.getc();

            // If the user types 'm' and we're not in manual mode already, switch modes.
            if (!manualMode && c == 'm') {
                manualMode = true;
                pc.printf("\r\nManual coordinate entry mode activated.\r\n");
                pc.printf("Enter coordinates manually:\r\n");
                continue; // Do not store the mode change character
            }

            // Store character in buffer if there's room
            if (index < buffer_size - 1) {
                buffer[index++] = c;
            } else {
                // If overflow, clear the buffer and inform the user
                pc.printf("\r\nBuffer overflow. Clearing buffer...\r\n");
                index = 0;
                continue;
            }

            // Check if the message is complete when newline or carriage return is received
            if (c == '\r' || c == '\n') {
                buffer[index] = '\0';  // Null-terminate the string

                // If any data was received, echo it back
                if (index > 0) {
                    if (manualMode) {
                        pc.printf("Manually entered coordinates: %s\r\n", buffer);
                    } else {
                        pc.printf("Coordinates received: %s\r\n", buffer);
                    }
                }
                index = 0;  // Reset the buffer for the next message
            }
        }
    }
}
