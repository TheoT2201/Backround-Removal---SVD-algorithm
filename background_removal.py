import cv2
import numpy as np

# Define SVD function
def svd(matrix):
    transpose = matrix.T @ matrix
    eigen_values, eigen_vectors = np.linalg.eig(transpose)
    V = eigen_vectors
    S = np.sqrt(eigen_values)
    U = matrix @ V @ np.linalg.inv(np.diag(S))
    return U, S, V.T


# Open the video file
cap = cv2.VideoCapture("dansator.mp4")

# Get the frames per second (fps) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Get the total number of frames in the video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize a background frame
bg_frame = None

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    
    # If there are no more frames, break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform SVD on the frame
    U, S, V = svd(gray)

    # Reconstruct the frame using the first k singular values
    k = 400
    recon = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
    recon = np.uint8(recon)

    # Update the background frame
    if bg_frame is None:
        bg_frame = recon
    else:
        bg_frame = (recon + bg_frame) // 2

    # Subtract the background from the current frame
    fg_frame = cv2.absdiff(gray, bg_frame)

    # Show the current frame
    cv2.imshow("Video", fg_frame)

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
