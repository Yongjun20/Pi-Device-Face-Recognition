# Face Recognition System

This project is a comprehensive face recognition system designed for Raspberry Pi, featuring a graphical user interface (GUI) built with Tkinter. It utilizes OpenCV for face detection, the Face_recognition library for encoding and matching faces, and integrates with external services for logging and data synchronization.

## Features

- **Face Capture**: Capture faces using the Raspberry Pi camera with manual or auto-capture modes. Supports adding or replacing existing face data for individuals.
- **Model Training**: Train the face recognition model on captured images, generating encodings stored in a pickle file for efficient recognition.
- **Real-Time Recognition**: Perform live face recognition, identifying known individuals and logging events. Includes confidence scoring and handling of unknown faces.
- **GUI Interface**: User-friendly Tkinter-based interface for managing capture, training, and recognition processes. Includes login system with admin privileges.
- **Data Synchronization**: Sync user data and logs with a mounted PC share for backup and centralized management.
- **SOAP Integration**: Communicate with a SOAP web service for additional functionality, such as user validation or data submission.
- **Logging**: Automatic logging of recognition events to local and server files, with timestamps.
- **Admin Management**: Manage admin users and permissions through the GUI.

## Requirements

- **Hardware**: Raspberry Pi with camera module (PiCamera2 compatible).
- **Software**:
  - Python 3.7+
  - OpenCV (opencv-python)
  - face_recognition library
  - picamera2 for camera access
  - Tkinter (usually included with Python)
  - Pillow (Pillow) for image processing
  - cryptography for encryption
  - 
Requests for HTTP requests
  - 
Numpy (dependency of face_recognition)

Install dependencies using pip:

`bash
pip install opencv-python face-recognition picamera2 pillow cryptography requests numpy
`

For Raspberry Pi, ensure PiCamera2 is properly installed and configured.

## Installation

1. Clone or download the project files to your Raspberry Pi.
2. Install the required Python packages as listed above.
3. Ensure the camera is connected and accessible.
4. Mount the PC share if using synchronization features (e.g., via /mnt/pcshare).
5. Run the application.

## Usage

1. **Start the Application**: Run python gui.py to launch the GUI.
2. **Login**: Use the login feature to access admin functions if needed.
3. **Capture Faces**:
   - Select a person name.
   - Choose capture mode (add or replace).
   - Use manual capture or enable auto-capture.
4. **Train Model**: Click the train button to process captured images and update the encodings file.
5. **Start Recognition**: Begin real-time recognition, which will identify faces and log events.
6. **View Logs**: Check local logs in the LOG folder or synced server logs.

## File Structure

- capture.py: Handles face capture from camera, face detection, and image saving.
- gui.py: Main application file with Tkinter GUI, threading for background tasks, and integration with other modules.
- 
ecog.py: Manages face recognition, encoding loading, and event handling/logging.
- train.py: Processes dataset images to generate face encodings and save to pickle file.
- README.md: This documentation file.
- dataset/: Folder containing subfolders for each person's face images.
- encodings.pickle: Serialized file containing trained face encodings and metadata.
- user.json: Local user data file, synced with server.
- admins.txt: List of admin users.
- settings.json: Application settings.
- LOG/: Folder for log files.

## Configuration

- **Camera**: Configured for 640x480 resolution. Adjust in capture.py if needed.
- **Face Detection**: Uses Haar cascade classifier from OpenCV.
- **Server Integration**: SOAP URL and server path are hardcoded; modify in code for different environments.
- **Encryption**: Uses Fernet for secure data handling; key is hardcoded (replace with generated key for security).

## Troubleshooting

- **Camera Issues**: Ensure PiCamera2 is installed and camera is not busy. Check permissions.
- **Face Detection Failures**: Verify Haar cascade path and OpenCV installation.
- **Recognition Accuracy**: Ensure good lighting and clear faces during capture and recognition.
- **Sync Errors**: Check mount point /mnt/pcshare and permissions for server access.
- **Dependencies**: Use virtual environments to avoid conflicts.

## License

This project is open-source. Modify and distribute as needed, but ensure compliance with library licenses (e.g., OpenCV, face_recognition).

## Contributing

Feel free to submit issues or pull requests for improvements.
