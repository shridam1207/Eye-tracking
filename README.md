# Pupil tracking based room automation system
Ever got ready with those big packs of popcorn and cold drinks, sat down comfortably on your sofa for a Netflix marathon or maybe, you just went to your comfy bed after a hard day’s work, but just realized you forgot to turn off the lights? What if you could control those annoying lights or maybe turn on your heater while you stay comfy in your bed with just your gaze? Thanks to the rise of IoT and advancements in Computer Vision, these ideas are no longer just fantasies. We have developed a system to control devices using just our gaze, completely based on open-source systems and your friendly neighborhood language, Python.
# The Idea
We first determine the user’s face landmarks(like lips, eyes, nose edges) to determine the pupil location using a webcam facing the user. Another webcam facing forward gives the user’s field of view. By mapping the pupil location we get the user’s current region of focus. We then determine the IoT device location and when it’s location and region of focus align for more than a second, we consider it as a signal to flip the switch.

NB: Most of the detailed explanations are given in the comments in the code

So, the whole project can be divided into the following steps:

1. Pupil tracking and region of focus
2. Device detection and recognition
3. Control signal
# Preliminaries
The project would need the following libraries to be installed-

Imutils – provides functions for easy image processing operations
OpenCV and OpenCV-contrib module – probably the best computer vision library available for Python
Numpy – package for scientific computation in Python
Dlib – toolkit for machine learning and data analysis applications
PySerial – library to communicate with the serial ports(here with Arduino)
Installation can be easily done using pip since all of them are available on PyPI
