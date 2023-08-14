import numpy as np
import h5py




class NamedArray(np.ndarray):
    def __new__(cls, data, index=None, columns=None, dtype=None):
        # Convert input data to an ndarray instance
        obj = np.asarray(data, dtype=dtype).view(cls)
        return obj

    def __init__(self, data, index=None, columns=None, dtype=None):
        # Set the axis labels
        self._set_axis_labels(0, index if index else list(range(self.shape[0])))
        self._set_axis_labels(1, columns if columns else list(range(self.shape[1])))

    def _set_axis_labels(self, axis, labels):
        """Set axis labels, ensure they are strings and unique."""
        if any(not isinstance(label, str) for label in labels):
            raise ValueError("All labels must be strings.")
        
        if len(labels) != len(set(labels)):
            raise ValueError("Labels must be unique.")
        
        if axis == 0 and len(labels) != self.shape[0]:
            raise ValueError(f"Number of row labels {len(labels)} does not match number of rows {self.shape[0]}.")
        elif axis == 1 and len(labels) != self.shape[1]:
            raise ValueError(f"Number of column labels {len(labels)} does not match number of columns {self.shape[1]}.")
        
        if axis == 0:
            self.row_labels = labels
        elif axis == 1:
            self.col_labels = labels

    def __repr__(self):
        max_rows, max_cols = 10, 10  # Maximum rows and columns to display before truncating
        data_str_list = []

        # Decide if truncation is needed
        truncate_rows = self.shape[0] > max_rows
        truncate_cols = self.shape[1] > max_cols

        rows_to_display = range(min(self.shape[0], max_rows))
        cols_to_display = range(min(self.shape[1], max_cols))

        # Constructing the displayed data string
        for i in rows_to_display:
            row_data = [f"{self[i, j]:.4f}" for j in cols_to_display]
            if truncate_cols:
                row_data.append("...")
            data_str_list.append(f"{self.row_labels[i]:>5} | " + " ".join(row_data))

        col_labels_to_display = [self.col_labels[j] for j in cols_to_display]
        if truncate_cols:
            col_labels_to_display.append("...")
        
        col_header = "      | " + " ".join(col_labels_to_display)
        data_str = "\n".join(data_str_list)

        return col_header + "\n" + data_str

    def get_labels(self, axis):
        """Return the labels for the given axis."""
        return self.row_labels if axis == 0 else self.col_labels

    def set_labels(self, axis, labels):
        """Set the labels for the given axis."""
        self._set_axis_labels(axis, labels)

    def _get_indices(self, key):
        """Convert named indices to integer indices."""
        if not isinstance(key, tuple):
            key = (key, slice(None, None, None))  # if only one key is provided, assume it's for rows
        
        indices = []
        labels_list = [self.row_labels, self.col_labels]
        for k, labels in zip(key, labels_list):
            if isinstance(k, list):
                idx_list = [labels.index(item) for item in k]
                indices.append(idx_list)
            elif isinstance(k, slice):
                start = labels.index(k.start) if k.start else None
                stop = labels.index(k.stop) if k.stop else None
                step = k.step
                indices.append(slice(start, stop, step))
            elif k in labels:
                indices.append(labels.index(k))
            else:
                indices.append(k)
        return tuple(indices)

    def __getitem__(self, key):
        return self.data[self._get_indices(key)]

    def __setitem__(self, key, value):
        self.data[self._get_indices(key)] = value
    
    def __add__(self, other):
        """Addition operation ensuring axes are aligned."""
        if self.row_labels != other.row_labels or self.col_labels != other.col_labels:
            raise ValueError("Axes are not aligned.")
        return NamedArray(self.data + other.data, index=self.row_labels, columns=self.col_labels)

    def __mul__(self, other):
        """Element-wise multiplication ensuring axes are aligned."""
        if self.row_labels != other.row_labels or self.col_labels != other.col_labels:
            raise ValueError("Axes are not aligned.")
        return NamedArray(self.data * other.data, index=self.row_labels, columns=self.col_labels)


    def __matmul__(self, other):
        """Matrix multiplication ensuring axes are aligned."""
        if self.col_labels != other.row_labels:
            raise ValueError("Axes are not aligned for matrix multiplication.")
        return NamedArray(np.dot(self.data, other.data), index=self.row_labels, columns=other.col_labels)
    
    def to_hdf(self, filename):
        """Save the NamedArray to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data', data=self.data)
            f.create_dataset('row_labels', data=self.row_labels, dtype=h5py.string_dtype())
            f.create_dataset('col_labels', data=self.col_labels, dtype=h5py.string_dtype())
    
    @classmethod
    def from_hdf(cls, filename):
        """Load a NamedArray from an HDF5 file."""
        with h5py.File(filename, 'r') as f:
            data = f['data'][:]
            row_labels = [label.decode('utf-8') for label in f['row_labels']]
            col_labels = [label.decode('utf-8') for label in f['col_labels']]
        return cls(data, index=row_labels, columns=col_labels)

