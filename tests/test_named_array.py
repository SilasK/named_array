import unittest
import numpy as np
from named_array import NamedArray

class TestNamedArray(unittest.TestCase):

    def setUp(self):
        self.data1 = np.array([[1, 2], [3, 4]])
        self.data2 = np.array([[1, 2], [3, 5]])
        self.row_labels1 = ['a', 'b']
        self.col_labels1 = ['x', 'y']
        self.row_labels2 = ['a', 'c']
        self.col_labels2 = ['x', 'z']

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


    def test_equal_data_and_labels(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        arr2 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        self.assertTrue(arr1 == arr2)

    def test_numpy_array_equal(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        arr2 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        np.testing.assert_array_equal(arr1.data,arr2.data)

        np.testing.assert_array_equal(arr1.__array__(),arr2.__array__())


    def test_different_data_same_labels(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        arr2 = NamedArray(self.data2, index=self.row_labels1, columns=self.col_labels1)
        self.assertFalse(arr1 == arr2)

    def test_same_data_different_row_labels(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        arr2 = NamedArray(self.data1, index=self.row_labels2, columns=self.col_labels1)
        self.assertFalse(arr1 == arr2)

    def test_same_data_different_col_labels(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        arr2 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels2)
        self.assertFalse(arr1 == arr2)

    def test_different_data_and_labels(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        arr2 = NamedArray(self.data2, index=self.row_labels2, columns=self.col_labels2)
        self.assertFalse(arr1 == arr2)

    def test_comparison_with_non_namedarray(self):
        arr1 = NamedArray(self.data1, index=self.row_labels1, columns=self.col_labels1)
        self.assertFalse(arr1 == self.data1)



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

        common_labels= dict(index=["a", "b"], columns=["x", "y"])

        arr1 = NamedArray([[1, 2], [3, 4]], **common_labels)
        arr2 = NamedArray([[2, 3], [4, 5]], **common_labels)

        
        # Addition
        result = arr1 + arr2

        np.testing.assert_array_equal(result.data, [[3, 5], [7, 9]])
        self.assertEqual(result.row_labels, ["a", "b"])
        self.assertEqual(result.col_labels, ["x", "y"])

        # Multiplication
        result = arr1 * arr2
        np.testing.assert_array_equal(result.data, [[2, 6], [12, 20]])
        self.assertEqual(result.row_labels, ["a", "b"])
        self.assertEqual(result.col_labels, ["x", "y"])

    def test_transpose(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        row_labels = ['a', 'b']
        col_labels = ['x', 'y', 'z']

        arr = NamedArray(data, index=row_labels, columns=col_labels)
        transposed_arr = arr.transpose()

        # Check if the data is transposed
        np.testing.assert_array_equal(transposed_arr.data, data.T)

        # Check if row and col labels are swapped
        self.assertListEqual(transposed_arr.row_labels, col_labels)
        self.assertListEqual(transposed_arr.col_labels, row_labels)

    def test_matrix_multiplication(self):
        """
        This test does the following:

        It sets up two NamedArray objects with known data and labels.
        It multiplies the two arrays together using the @ operator.
        It checks if the resulting data is correct (based on known expected results).
        It verifies that the resulting labels are correctly set.
        Finally, it also tests the error handling for the matrix multiplication operation by attempting to multiply two arrays with misaligned labels.
        The multiplication should raise a ValueError in this case.
        
        
        """
        data1 = np.array([[1, 2], [3, 4]])
        row_labels1 = ['a', 'b']
        col_labels1 = ['x', 'y']

        data2 = np.array([[5, 6], [7, 8]])
        row_labels2 = ['x', 'y']
        col_labels2 = ['i', 'j']

        arr1 = NamedArray(data1, index=row_labels1, columns=col_labels1)
        arr2 = NamedArray(data2, index=row_labels2, columns=col_labels2)

        result = arr1 @ arr2

        # Check if the matrix multiplication is correct
        expected_data = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result.data, expected_data)

        # Check if the resulting labels are correct
        self.assertListEqual(result.row_labels, row_labels1)
        self.assertListEqual(result.col_labels, col_labels2)

        # Test error handling for misaligned matrix multiplication
        data3 = np.array([[1, 2], [3, 4]])
        row_labels3 = ['a', 'b']
        col_labels3 = ['k', 'l']

        arr3 = NamedArray(data3, index=row_labels3, columns=col_labels3)
        with self.assertRaises(ValueError):
            _ = arr1 @ arr3

    def test_label_preservation(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        row_labels = ['a', 'b', 'c']
        col_labels = ['x', 'y', 'z']

        arr = NamedArray(data, index=row_labels, columns=col_labels)

        # Testing mean along axis 0 (columns)
        col_mean = np.mean(arr, axis=0)
        self.assertListEqual(col_mean.col_labels, col_labels)

        # Testing mean along axis 1 (rows)
        row_mean = np.mean(arr, axis=1)
        self.assertListEqual(row_mean.row_labels, row_labels)

        # Testing min along axis 0 (columns)
        col_min = np.min(arr, axis=0)
        self.assertListEqual(col_min.col_labels, col_labels)

        # Testing min along axis 1 (rows)
        row_min = np.min(arr, axis=1)
        self.assertListEqual(row_min.row_labels, row_labels)

        # Testing max along axis 0 (columns)
        col_max = np.max(arr, axis=0)
        self.assertListEqual(col_max.col_labels, col_labels)

        # Testing max along axis 1 (rows)
        row_max = np.max(arr, axis=1)
        self.assertListEqual(row_max.row_labels, row_labels)



    # def test_serialization(self):
    #     arr = NamedArray([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        
    #     # Saving to HDF
    #     file_path = "test_named_array.hdf5"
    #     arr.to_hdf(file_path)
        
    #     # Loading from HDF
    #     loaded_arr = NamedArray.from_hdf(file_path)
        
    #     # Checking if the loaded data matches the original
    #     np.testing.assert_array_equal(loaded_arr, arr)
    #     self.assertEqual(loaded_arr.row_labels, arr.row_labels)
    #     self.assertEqual(loaded_arr.col_labels, arr.col_labels)

if __name__ == '__main__':
    unittest.main()