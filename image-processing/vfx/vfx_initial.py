import cv2
import numpy as np

def main(source_video_path, background_image_path, output_video_path):
    # Initialize video capture for source video
    cap = cv2.VideoCapture(source_video_path)
    # if not cap.isOpened():
    #     print("Error: Could not open source video")
    #     return

    # Capture the background from the background image
    background = cv2.imread(background_image_path)
    # if background is None:
    #     print("Error: Could not open background image")
    #     return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ensure background matches the video dimensions
    background = cv2.resize(background, (width, height))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define the color range for detecting the green screen
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    print("Processing video. Please wait...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask to detect the green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Refine the mask using morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Create an inverse mask to segment out the green screen from the frame
        mask_inv = cv2.bitwise_not(mask)

        # Segment the green screen out of the frame using bitwise AND with the inverse mask
        res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Segment the background region using the mask
        res2 = cv2.bitwise_and(background, background, mask=mask)

        # Combine the two results to get the final output
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

        # Write the frame to the output video
        out.write(final_output)


    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to", output_video_path)

if __name__ == "__main__":
    source_video_path = r"C:\Users\Chirag\OneDrive\Documents\Code\dlwp-training\image-processing\vfx\videoplayback.mp4"
    background_image_path = r"C:\Users\Chirag\OneDrive\Documents\Code\Training\images\lionel-messi.jpg"
    output_video_path = "output_video.avi"
    main(source_video_path, background_image_path, output_video_path)
