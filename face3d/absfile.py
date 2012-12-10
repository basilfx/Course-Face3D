"""
Copyright (c) 2012 Bas Stottelaar, Jeroen Senden
See the file LICENSE for copying permission.
"""

import os
import numpy

class AbsFile(object):
    # Class variables
    invalid_value = -999999

    def __init__(self, filename):
        # Check if file exists
        if not os.path.exists(filename):
            raise Exception("Data file does not exist")

        # Create variables
        self.data = {}
        self.row_size = False
        self.col_size = False

        # Now read the file and create an XYZ matrix
        with open(filename, 'r') as file_handle:
            # Helper for dimension
            def read_dimension(dimension):
                line = file_handle.readline()
                data = line.strip().split(" ")

                if len(data) == 2 and data[1] == dimension:
                    return int(data[0])
                else:
                    raise Exception("Invalid header: expected '%s'" % dimension)

            # Helper for data order
            def read_data_type():
                line = file_handle.readline()
                data = line.strip().split(" ", 1)

                if len(data) == 2 and data[0] == "pixels":
                    return data[1][1:-2].split(" ")
                else:
                    raise Exception("Invalid header: expected data type")

            # Helper for reading data lines
            def read_data(data_type):
                # Initialize result array
                data_type = numpy.int if data_type == 'flag' else numpy.float
                result = numpy.zeros(self.col_size * self.row_size, dtype=data_type)
                index = 0

                # Read line
                line = file_handle.readline()
                data = line.strip().split(" ")

                for value in data:
                    try:
                        # Convert string to correct format
                        result[index] = data_type(value)

                        # Increment index
                        index = index + 1
                    except ValueError:
                        print "Unexpected input: expected '%s', got '%s'" % (data_type, value)

                # In case of invalid values, mask invalid values
                if data_type == numpy.float:
                    result[result == AbsFile.invalid_value] = numpy.nan

                # Return reshaped array
                return result.reshape(self.row_size, self.col_size)

            # Read the header of the file
            self.row_size = read_dimension('rows')
            self.col_size = read_dimension('columns')
            self.data_type = read_data_type()

            # Read the actual data
            for current_data_type in self.data_type:
                self.data[current_data_type] = read_data(current_data_type)

    @property
    def width(self):
        return self.col_size

    @property
    def height(self):
        return self.row_size