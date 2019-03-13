# cuda-lane-detection

# Compiling:
g++ -I/usr/local/include/opencv4 -L/usr/local/lib/ HoughTransform.cpp LaneDetection.cpp Preprocessing.cpp Line.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -std=c++11 -o lanedetect

nvcc -I/usr/local/include/opencv4 -L/usr/local/lib/ HoughTransform.cu LaneDetection.cpp Preprocessing.cpp Line.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -std=c++11 -o lanedetect

# Running:
./lanedetect lanes.png --seq

./lanedetect lanes.png
