import cv2
import numpy as np

# Define SVD function
def svd(A):
    m, n = A.shape
    Q = np.eye(n)
    R = A
    for _ in range(10*n**2):
        Q_new, R_new = np.zeros((n,n)), np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                R_new[i,j] = np.dot(Q[:,i], R[:,j])
            for j in range(n):
                Q_new[:,j] += R_new[i,i] * Q[:,j]
            Q_new[:,i] = Q_new[:,i] / np.linalg.norm(Q_new[:,i])
            for j in range(i+1, n):
                R_new[i,j] = R_new[i,j] / np.dot(Q_new[:,i], Q[:,j])
        Q = Q_new
        R = R_new
    S = np.diag(R)
    V = Q
    U = np.dot(A,np.linalg.inv(np.diag(S)))
    return U, S, V

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
    k = 50
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
