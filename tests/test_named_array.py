import unittest
import numpy as np
from named_array import NamedArray

class TestNamedArray(unittest.TestCase):

    def test_initialization(self):
        # Basic initialization with default labels
        arr = NamedArray([[1, 2], [3, 4]])
        self.assertEqual(arr.row_labels, ["0", "1"])
        self.assertEqual(arr.col_labels, ["0", "1"])

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
        self.assertEqual(arr.get_labels(0), ["0", "1"])
        self.assertEqual(arr.get_labels(1), ["0", "1"])

        # Testing set_labels
        arr.set_labels(0, ["a", "b"])
        arr.set_labels(1, ["x", "y"])
        self.assertEqual(arr.row_labels, ["a", "b"])
        self.assertEqual(arr.col_labels, ["x", "y"])

    def test_indexing(self):

        values = np.array([[1, 2, 3, 4], 
                          [5, 6 ,7, 8],
                          [9,10,11,12]])
        
        arr = NamedArray(values, index=["S1", "S2","S3"], columns=["a", "b","c","d"])

        # Integer indexing
        self.assertEqual(arr[0, 1], 2)

        # Label-based indexing
        self.assertEqual(arr["S1", "a"], 1)
        self.assertEqual(arr["S2", "b"], 6)
        

        
        # Mixed indexing
        self.assertEqual(arr["S1", 1], 2)
        self.assertEqual(arr[0, "c"], 3)

        #subseting

        self.assertEqual(arr["S1", 1], 2)

        # Slicing
        sub_arr = arr[ "S1":"S3","a":"c"]
        solution= np.array([[1, 2], [5, 6]])

        self.assertEqual(sub_arr.shape, (2, 2))
        self.assertEqual(sub_arr.row_labels, ["S1", "S2"])
        self.assertEqual(sub_arr.col_labels, ["a", "b"])

        np.testing.assert_array_equal(sub_arr.data,solution )



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
        result = arr1 @ arr2
        np.testing.assert_array_equal(result, [[10, 13], [22, 29]])

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