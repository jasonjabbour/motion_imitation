from pyfirmata import Arduino, SERVO, util
import time

port = 'COM7'
pin = 8
board = Arduino(port)

board.digital[pin].mode = SERVO

def rotate_servo(pin, angle):
    board.digital[pin].write(angle)
    time.sleep(.015)

while True:
    x = input("input: ")
    if x=="1":
        for i in range(0,90):
            rotate_servo(pin,i)
    elif x=="2":
        for i in range(0,180):
            rotate_servo(pin,i)
    elif x=="3":
        for i in range(0,270):
            rotate_servo(pin,i)


# servo control 15.12.2016
 
# 1) user set servo position in python
# 2) position is sent to arduino
# 3) arduino moves servo to position
# 4) arduino send confirmation message back to python
# 5) message is printed in python console
 
# import serial                                           # import serial library
# arduino = serial.Serial('/dev/cu.usbmodem1421', 9600)   # create serial object named arduino
# while True:                                             # create loop
 
#         command = str(input ("Servo position: "))       # query servo position
#         arduino.write(command)                          # write position to serial port
#         reachedPos = str(arduino.readline())            # read serial port for arduino echo
#         print(reachedPos)                               # print arduino echo to console