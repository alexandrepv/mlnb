import numpy as np
import time


def example_1():

    pass

def example_2():

    def measure_access_time(matrix_2d: np.ndarray, epochs=100):

        t0 = time.perf_counter()
        for i in range(epochs):
            for j in range(matrix_2d.shape[0]):
                result = data[j, :] * 3.14
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        for i in range(epochs):
            for j in range(matrix_2d.shape[1]):
                result = data[:, j] * 3.14
        t3 = time.perf_counter()

        print(f' > Row access: {t1 - t0:.3f} seconds')
        print(f' > Columns access: {t3 - t2:.3f} seconds')

    # Create row-major matrix for demo
    data = np.random.rand(1000, 1000).astype(np.float32)

    # Measure
    print(f'\n[ Row Major Matrix ]')
    measure_access_time(matrix_2d=data)

    # Now, change memory layout from row-major to column-major
    data = np.asfortranarray(data)

    # Measure again
    print(f'\n[ Column Major Matrix ]')
    measure_access_time(matrix_2d=data)




if __name__ == "__main__":

    example_2()
