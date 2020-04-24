# Rock Pi N10 (RK3399Pro) NPU demo


### Follow links below to install prerequisites.

https://wiki.radxa.com/RockpiN10/hardware/camera/

https://wiki.radxa.com/AI/RKNN-Toolkit

### Clone
$ git clone https://github.com/metanav/RockPiN10.git

$ cd RockPiN10/rknn_tflite_mobilenet_v1

Note: change the host and port to receive UDP stream at your target machine at line 127 in the inference.py script.
The host ip is your desktop or laptop ip address.
### Run
$ python3 inference.py

At the target machine install Gstreamer and required plugins (I used this command to install on my Mac; brew install gstreamer gst-plugins-good gst-plugins-bad) and run the following command (change port to match with the source above):
$ gst-launch-1.0 udpsrc port=1234  ! application/x-rtp,encoding-name=JPEG,payload=26  ! rtpjpegdepay ! jpegdec ! queue ! autovideosink



