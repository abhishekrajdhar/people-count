👁️‍🗨️ People Counting using Object Tracking (OpenCV + MobileNet-SSD)

This project uses OpenCV and a pre-trained MobileNet-SSD deep learning model to detect and track people in video frames.
It also counts the number of people moving up or down across a virtual line drawn in the video.

The output is saved as a processed video with bounding boxes, tracking IDs, and up/down counters displayed on each frame.

📸 Project Overview

This system detects and tracks people in a given video file using the MobileNet-SSD deep learning architecture.
A custom Centroid Tracker is implemented to assign unique IDs to people and maintain their identity across frames.

Whenever a person crosses the reference line (usually the center of the frame), the system updates the Up or Down count.

🎯 Features

Real-time object detection and tracking

Counts people moving up and down

Displays bounding boxes, object IDs, and movement direction

Saves processed video as output (output_people_count.mp4)

Handles disappearing and reappearing objects using a robust Centroid Tracker

🧠 Working Principle

Detection:
The MobileNet-SSD model detects all objects in the frame and filters only those labeled as "person".

Tracking:
The Centroid Tracker assigns a unique ID to each detected person and updates their position frame by frame using centroid distances.

Counting:
A horizontal reference line is drawn across the frame.

If a person crosses it from bottom → top, the Up Count increases.

If a person crosses it from top → bottom, the Down Count increases.

Output:
The processed frames are written into an output video file with visual annotations and final counts.

🧩 Project Structure
├── deploy.prototxt                 # MobileNet-SSD model architecture
├── mobilenet_iter_73000.caffemodel # Pre-trained weights
├── test.mp4                        # Input video file
├── output_people_count.mp4         # Processed output video
├── people_counter.py               # Main project script
└── README.md                       # Project documentation

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/abhishekrajdhar/people-count.git
cd people-count

2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed. Then install the required packages:

pip install opencv-python numpy

▶️ Usage
Run the Script
python main.py

Controls

Press q to quit the video window early.

The processed video will be saved automatically as output_people_count.mp4.

🧮 Parameters
Parameter	Description	Default
max_disappeared	Number of frames to wait before deregistering a lost object	50
confidence threshold	Minimum confidence score to consider a detection valid	0.5
line_position	Y-coordinate of the counting line	Mid of frame height
📊 Example Output

Detected Objects:

Bounding boxes around each person

Unique ID labels

Centroid dots (blue)

Up/Down counts displayed on the top-left

Console Output:

Person 2 moved up.
Person 5 moved down.
Person 3 moved up.

🧠 Model Information

The project uses the MobileNet SSD (Single Shot MultiBox Detector) model trained on the COCO dataset.
It supports multiple object classes, but here we only use the "person" class (class ID 15).

Model files:

deploy.prototxt → Network architecture definition

mobilenet_iter_73000.caffemodel → Pre-trained weights

🧾 Future Enhancements

Replace MobileNet-SSD with YOLOv8 or OpenVINO for faster detection

Add bi-directional counting accuracy optimization

Implement zone-based tracking (multi-line counting)

Integrate with a web dashboard for analytics visualization

🧑‍💻 Author

Abhishek Dubey
📧 rajdhardubey4434@gmail.com

🏁 Final Output Preview

✅ The final output video (output_people_count.mp4) will contain live bounding boxes, person IDs, and counters showing people moving up and down.
