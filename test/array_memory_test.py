# License: BSD 3 clause

import gc, unittest, weakref

import numpy as np
import scipy
from scipy.sparse import csr_matrix

import statick

class Test(unittest.TestCase):

    def test_s_sparse_array2d_memory_leaks(self):
        """...Test brute force method in order to see if we have a memory leak
        during typemap out
        """
        import os

        def deserialize(file):
            return
            # cap = statick.load_double_sparse2d(file)
            # ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.py_object
            # ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            # ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
            # ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
            # name = ctypes.pythonapi.PyCapsule_GetName(cap)
            # return (cap, ctypes.pythonapi.PyCapsule_GetPointer(cap, name))

        try:
            import psutil
        except ImportError:
            print('Without psutils we cannot ensure we have no memory leaks')
            return


        def get_memory_used():
            """Returns memory used by current process
            """
            process = psutil.Process(os.getpid())
            return process.memory_info()[0]

        cereal_file = "sparse.gen.cereal"
        try:
            n_rows = int(1e3)
            n_cols = int(1e2)
            s_spar = int((n_rows * n_cols) * .3)
            data_size = (s_spar * 8)
            row_size = (n_rows + 1) * 8
            # The size in memory of an array of ``size`` doubles
            bytes_size = (data_size * 2) + (row_size)

            sparsearray_double = scipy.sparse.rand(
                n_rows, n_cols, 0.3, format="csr", dtype=np.float64)

            statick.save_double_sparse2d(sparsearray_double, cereal_file)

            gc.collect()
            initial_memory = get_memory_used()
            a = statick.load_double_sparse2d(cereal_file)
            first_filled_memory = get_memory_used()

            # Check that new memory is of the correct order (10%)
            self.assertAlmostEqual(first_filled_memory - initial_memory,
                                   bytes_size, delta=1.1 * bytes_size)
            a = 1
            for i in range(10):
                # Check memory is not increasing
                gc.collect()
                filled_memory = get_memory_used()
                self.assertAlmostEqual(filled_memory, initial_memory,
                                       delta=1.1 * bytes_size)
                X = statick.load_double_sparse2d(cereal_file)
                X = 1
            gc.collect()
            end = get_memory_used()
            self.assertAlmostEqual(end, initial_memory, delta=1.1 * bytes_size)

        finally:
            if os.path.exists(cereal_file):
                os.remove(cereal_file)



if __name__ == "__main__":
    unittest.main()
