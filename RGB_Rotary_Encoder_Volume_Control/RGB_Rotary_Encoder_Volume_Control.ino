// "RGB_Rotary_Encoder"

// HARDWARE CONNECTIONS

// Rotary encoder pin A to digital pin 3*
// Rotary encoder pin B to analog pin 3
// Rotary encoder pin C to ground

// This sketch implements software debounce, but you can further
// improve performance by placing 0.1uF capacitors between
// A and ground, and B and ground.

// Rotary encoder pin 1 (red cathode) to digital pin 10
// Rotary encoder pin 2 (green cathode) to analog pin 1
// Rotary encoder pin 3 (button) to digital pin 4
// Rotary encoder pin 4 (blue cathode) to digital pin 5
// Rotary encoder pin 5 (common anode) to VCC (3.3V or 5V)

// Note that because this is a common anode device,
// the pushbutton requires an external 1K-10K pullDOWN resistor
// to operate.

// * Pins marked with an asterisk should not change because
// they use interrupts on that pin. All other pins can change,
// see the constants below.

// SERIAL MONITOR

// Run this sketch with the serial monitor window set to 9600 baud.

// HOW IT WORKS

// The I/O pins used by the rotary encoder hardware are set up to
// automatically call interrupt functions (rotaryIRQ and buttonIRQ)
// each time the rotary encoder changes states.

// The rotaryIRQ function transparently maintains a counter that
// increments or decrements by one for each detent ("click") of
// the rotary encoder knob. This function also sets a flag
// (rotary_change) to true whenever the counter changes. You can
// check this flag in your main loop() code and perform an action
// when the knob is turned.

// The buttonIRQ function does the same thing for the pushbutton
// built into the rotary encoder knob. It will set flags for
// button_pressed and button_released that you can monitor in your
// main loop() code. There is also a variable for button_downtime
// which records how long the button was held down.

// There is also code in the main loop() that keeps track
// of whether the button is currently being held down and for
// how long. This is useful for "hold button down for five seconds
// to power off"-type situations, which cannot be handled by
// interrupts alone because no interrupts will be called until
// the button is actually released.

#include <Bounce.h>
#include <Encoder.h>

#define ROT_LEDG A8  // green LED
#define ROT_B 5      // rotary A
#define ROT_A 3      // rotary B
#define ROT_SW 4     // rotary puhbutton
#define ROT_LEDB A9  // blue LED
#define ROT_LEDR A5  // red LED
#define TEENSY_LED 13

volatile int rotary_counter = 0;  // current "position" of rotary encoder (increments CW)
volatile int volume = 0;

// // Change these two numbers to the pins connected to your encoder.
// //   Best Performance: both pins have interrupt capability
// //   Good Performance: only the first pin has interrupt capability
// //   Low Performance:  neither pin has interrupt capability
Encoder myEnc(ROT_B, ROT_A);

// Instantiate a Bounce object with a 5 millisecond debounce time
Bounce bouncer = Bounce(ROT_SW, 5);

void setup() {
  pinMode(TEENSY_LED, OUTPUT);
  digitalWrite(TEENSY_LED, 1);
  pinMode(ROT_SW, INPUT);
  // The rotary switch is common anode with external pulldown, do not turn on pullup
  pinMode(ROT_LEDB, OUTPUT);
  pinMode(ROT_LEDG, OUTPUT);
  pinMode(ROT_LEDR, OUTPUT);

  volumeLED(volume);

  Serial.begin(115200);  // Use serial for debugging
  Serial.println("Begin RGB Rotary Encoder Testing");
}

long lastPosition = 0;
const int stepsPerDetent = 2;  // Adjust based on your encoder (common values: 2 or 4)
int messageCounter = 0;        // Tracks how many messages have been processed
bool mute = false;

void loop() {
  // Update the debouncer
  bouncer.update();

  // Get the update value
  int value = bouncer.read();

  // Turn on or off the LED
  if ((value == HIGH) & (mute == false)) {
    Serial.println("Muted");
    setMUTE();
    mute = true;
    delay(1000);  // Small delay for stability
  } else if ((value == HIGH) & (mute == true)) {
    Serial.println("Unmuted");
    volumeLED(volume);
    mute = false;
    delay(1000);  // Small delay for stability
  } else if (volume == 0) {
    setMUTE();
  }

  if (mute == false) {
    long newPosition = myEnc.read();
    int movement = newPosition - lastPosition;

    if (abs(movement) >= stepsPerDetent) {
      messageCounter++;  // Increment the counter

      if (messageCounter % 2 == 1) {  // Only print every second message
        if (newPosition > lastPosition) {
          Serial.println("Rotating Counterclockwise ←");
          if (volume - 1 <= 0) {
            volume = 0;
          } else {
            volume = volume - 1;
          }
          Serial.println(volume);
          volumeLED(volume);
        } else {
          Serial.println("Rotating Clockwise →");
          if (volume + 1 > 50) {
            volume = 50;
          } else {
            volume = volume + 1;
          }
          Serial.println(volume);
          volumeLED(volume);
        }
      }

      lastPosition = newPosition;  // Update position
    }
  }
  delay(5);  // Small delay for stability
}

void setMUTE()
{
  analogWrite(ROT_LEDR, 0);
  analogWrite(ROT_LEDG, 255);
  analogWrite(ROT_LEDB, 255);
}

void volumeLED(int volume)  // Volume from 1 - 50
{
  analogWrite(ROT_LEDR, 0);
  //analogWrite(ROT_LEDG, 255 - round((97 / 50) * volume));
  analogWrite(ROT_LEDG, round(194 / (1 + exp(0.17 * volume)) + 158));
  //analogWrite(ROT_LEDB, round((255 / 50) * volume));
  analogWrite(ROT_LEDB, round(529.031 / (1 + exp(-0.08 * volume)) - 264.516));
}
