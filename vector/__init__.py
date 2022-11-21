"""
Creates a vector object which is intended to be used as a datatype.

__init__.py is initialised on library import and therefore the vector class should always
be used with this library and is therefore located here.

Raises:
    Exception: Error subtracting
    Exception: Error adding
    Exception: Error multiplying
    Exception: Error index
"""

class Vector():
    """
    Designed to create a vector object to be used by the main program

    Args:
        vector (list): the vector object

    Attributes:
        vector (list): the vector

    Methods:
        sum:
            Summates the whole list - important for machine learning

            Returns:
                float: the sum of all values within the list
    """

    def __init__(self, vector):
        self.__vector = list(vector) if isinstance(vector, (list,tuple)) else [vector]

    def __sub__(self, other):
        if isinstance(other, Vector) and len(self.__vector) == len(other):
            return Vector([float(self.__vector[i]) - float(other[i]) for i in range(len(self))])
        if isinstance(other, (int, float)):
            return Vector([(float(self[i]) - other) for i in range(len(self))])

        raise Exception(f"Error subtracting: \
could not subtract length {len(self.__vector)} from length {len(other)}")

    def __add__(self, other):
        if isinstance(other, Vector) and len(self.__vector) == len(other):
            return Vector([float(self.__vector[i]) + float(other[i]) for i in range(len(self))])
        if isinstance(other, (int, float)):
            return Vector([float(self.__vector[i]) + other for i in range(len(self))])

        raise Exception(f"Error adding: \
could not add length {len(self.__vector)} with length {len(other)}")

    def __mul__(self, other):
        if isinstance(other, Vector) and len(self.__vector) == len(other):
            sum_ans = 0
            for index, value in enumerate(self.__vector):
                sum_ans += float(value * other[index])
            return Vector(sum_ans)
        if isinstance(other, (int, float)):
            return Vector([float(self.__vector[i]) * other for i in range(len(self.__vector))])

        raise Exception(f"Error multiplying: \
could not multiply length {len(self.__vector)} with length {len(other)}")

    def __getitem__(self, index):
        if 0 <= index < len(self.__vector):
            return self.__vector[index]

        raise Exception(f"Error index: \
Invalid index {index} for vector with length {len(self.__vector)}")

    def __repr__(self):
        return f"{self.__vector}"

    def __iter__(self):
        return (i for i in self.__vector)

    def __len__(self):
        return len(self.__vector)

    def __str__(self):
        return f"{self.__vector}".replace("[", "").replace("]", "")

    def __float__(self):
        if len(self.__vector) != 1:
            raise Exception(f"Error converting: \
Cannot convert lenght {len(self.__vector)} to float.")

        return float(self.__vector[0])

    def __int__(self):
        if len(self.__vector) != 1:
            raise Exception(f"Error converting: \
Cannot convert lenght {len(self.__vector)} to int.")

        return int(self.__vector[0])

    def __eq__(self, other):
        return self.__vector == other

    def sum(self):
        """
        Summates the whole list - important for machine learning

        Returns:
            float: the sum of all values within the list
        """
        return sum(self.__vector)

    @property
    def vector(self):
        """
        Returns the private vector object

        Returns:
            list: the vector items
        """
        return self.vector()

    @vector.setter
    def vector(self, new_vector):
        """
        Sets the value of private object vector.

        Args:
            new_vector (_type_): _description_
        """
        self.__vector = new_vector
