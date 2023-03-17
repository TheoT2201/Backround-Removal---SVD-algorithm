import cv2
import numpy as np

# Capture the video
cap = cv2.VideoCapture("dansator.mp4")

def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for i in range(n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j].T, A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R

def svd_decomposition(A):
    m, n = A.shape
    Q, R = qr_decomposition(A)
    S = np.dot(Q.T, A)
    U, S, V = np.linalg.svd(S)
    return U, S, V

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute QR decomposition
    Q, R = qr_decomposition(gray)

    # Compute SVD decomposition
    U, S, V = svd_decomposition(R)

    # Select the number of singular values to keep
    k = 100
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vk = V[:k, :]

    # Reconstruct the original matrix
    Rk = Uk @ Sk @ Vk

    # Create the binary mask
    ret, mask = cv2.threshold(Rk, 150, 255, cv2.THRESH_BINARY)

    # Remove the background
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = cv2.convertScaleAbs(mask, dst=frame)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the resulting frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
