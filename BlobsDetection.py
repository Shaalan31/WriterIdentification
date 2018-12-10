from commonfunctions import *
import threading

areas = []
lengths = []
roundness = []
form_factors = []
lock1 = threading.Lock()
lock2 = threading.Lock()
lock3 = threading.Lock()
lock4 = threading.Lock()
num_threads = 5
step = num_threads


def blobs_features(contours, hierarchy):
    mask = (hierarchy[:, 3] > 0).astype('int')
    contours = contours[np.where(mask)]

    t1 = threading.Thread(target=loop_on_contours, args=(contours, 1))
    t2 = threading.Thread(target=loop_on_contours, args=(contours, 2))
    t3 = threading.Thread(target=loop_on_contours, args=(contours, 3))
    t4 = threading.Thread(target=loop_on_contours, args=(contours, 4))
    t5 = threading.Thread(target=loop_on_contours, args=(contours, 5))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    return [np.average(areas), np.average(roundness), np.average(form_factors)]


def loop_on_contours(contours, starting_index):
    for i in range(starting_index, len(contours), step):
        contour = contours[i]
        current_area = cv2.contourArea(contour)
        if current_area == 0:
            continue
        current_length = cv2.arcLength(contour, True)
        current_length_sq = current_length * current_length

        lock1.acquire()
        areas.append(current_area)
        lock1.release()

        lock2.acquire()
        lengths.append(current_length)
        lock2.release()

        lock3.acquire()
        form_factors.append(4 * current_area * math.pi / current_length_sq)
        lock3.release()

        lock4.acquire()
        roundness.append(current_length_sq / current_area)
        lock4.release()
