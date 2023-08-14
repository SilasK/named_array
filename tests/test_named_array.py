import unittest
import numpy as np
from named_array import NamedArray

class TestNamedArray(unittest.TestCase):

    def test_initialization(self):
        # Basic initialization with default labels
        arr = NamedArray([[1, 2], [3, 4]])
        self.assertEqual(arr.row_labels, [0, 1])
        self.assertEqual(arr.col_labels, [0, 1])

        # Initialization with custom labels
        arr = NamedArray([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        self.assertEqual(arr.row_labels, ["a", "b"])
        self.assertEqual(arr.col_labels, ["x", "y"])

        # Test for error with non-string labels
        with self.assertRaises(ValueError):
            NamedArray([[1, 2], [3, 4]], index=[1, 2], columns=["x", "y"])

        # Test for error with duplicate labels
        with self.assertRaises(ValueError):
            NamedArray([[1, 2], [3, 4]], index=["a", "a"], columns=["x", "y"])

    def test_label_management(self):
        arr = NamedArray([[1, 2], [3, 4]])
        
        # Testing get_labels
        self.assertEqual(arr.get_labels(0), [0, 1])
        self.assertEqual(arr.get_labels(1), [0, 1])

        # Testing set_labels
        arr.set_labels(0, ["a", "b"])
        arr.set_labels(1, ["x", "y"])
        self.assertEqual(arr.row_labels, ["a", "b"])
        self.assertEqual(arr.col_labels, ["x", "y"])

    def test_indexing(self):
        arr = NamedArray([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])

        # Label-based indexing
        self.assertEqual(arr["a", "x"], 1)
        self.assertEqual(arr["b", "y"], 4)
        
        # Integer indexing
        self.assertEqual(arr[0, 1], 2)
        
        # Mixed indexing
        self.assertEqual(arr["a", 1], 2)
        
        # Slicing
        sub_arr = arr["a":"b", "x":"y"]
        np.testing.assert_array_equal(sub_arr, [[1, 2], [3, 4]])

    def test_arithmetic_operations(self):
        arr1 = NamedArray([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        arr2 = NamedArray([[2, 3], [4, 5]], index=["a", "b"], columns=["x", "y"])

        # Addition
        result = arr1 + arr2
        np.testing.assert_array_equal(result, [[3, 5], [7, 9]])

        # Multiplication
        result = arr1 * arr2
        np.testing.assert_array_equal(result, [[2, 6], [12, 20]])

        # Matrix multiplication
        arr3 = NamedArray([[2], [3]], index=["x", "y"])
        result = arr1 @ arr3
        np.testing.assert_array_equal(result, [[8], [18]])

    def test_serialization(self):
        arr = NamedArray([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        
        # Saving to HDF
        file_path = "test_named_array.hdf5"
        arr.to_hdf(file_path)
        
        # Loading from HDF
        loaded_arr = NamedArray.from_hdf(file_path)
        
        # Checking if the loaded data matches the original
        np.testing.assert_array_equal(loaded_arr, arr)
        self.assertEqual(loaded_arr.row_labels, arr.row_labels)
        self.assertEqual(loaded_arr.col_labels, arr.col_labels)

if __name__ == '__main__':
    unittest.main()