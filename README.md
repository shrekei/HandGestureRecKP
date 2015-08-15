# HandGestureRecKP

This project is able to recognize the ten digit hand gestures with a commercial depth camera. This implementation is based on part of our work in the two papers [1, 2], which first uses the method in [1] to parse the hand into 12 different hand parts and extract the joints from the parsed hand parts, and then use the template matching method in [2] to recognize the hand gesture. The project is written in C++/OpenCV and runs in real-time.

[1] Hui Liang, Junsong Yuan and Daniel Thalmann, Parsing the Hand in Depth Images, in IEEE Trans. Multimedia, vol. 16, no. 5, Aug. 2014. [Video Link]

[2] Hui Liang and Junsong Yuan, Hand Parsing and Gesture Recognition with a Commodity Depth Camera, in Computer Vision and Machine Learning with RGB-D Sensors, Springer, 2014.
