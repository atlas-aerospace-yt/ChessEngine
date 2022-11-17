"""
Creates a vector object which is intended to be used as a datatype.

Raises:
    Exception: Error subtracting
    Exception: Error adding
    Exception: Error multiplying
    Exception: Error index
"""


class Vector():
    """
    Vector class

    Designed to create a vector object to be used by the main program
    """

    def __init__(self, vector):
        self.vector = vector if isinstance(vector, list) else [vector]

    def __sub__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            return Vector([float(self.vector[i]) - float(other[i]) for i in range(len(self))])
        if isinstance(other, (int, float)):
            return Vector([(float(self[i]) - other) for i in range(len(self))])

        raise Exception(f"Error subtracting: \
could not subtract length {len(self.vector)} from length {len(other)}")

    def __add__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            return Vector([float(self.vector[i]) + float(other[i]) for i in range(len(self))])
        if isinstance(other, (int, float)):
            return Vector([float(self.vector[i]) + other for i in range(len(self))])

        raise Exception(f"Error adding: \
could not add length {len(self.vector)} with length {len(other)}")

    def __mul__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            sum_ans = 0
            for index, value in enumerate(self.vector):
                sum_ans += float(value * other[index])
            return Vector(sum_ans)
        if isinstance(other, (int, float)):
            return Vector([float(self.vector[i]) * other for i in range(len(self.vector))])

        raise Exception(f"Error multiplying: \
could not multiply length {len(self.vector)} with length {len(other)}")

    def __getitem__(self, index):
        if 0 <= index < len(self.vector):
            return self.vector[index]

        raise Exception(f"Error index: \
Invalid index {index} for vector with length {len(self.vector)}")

    def __len__(self):
        return len(self.vector)

    def __str__(self):
        return f"{self.vector}".replace("[", "").replace("]", "")

    def __float__(self):
        return self.vector[0]

    def __int__(self):
        return self.vector[0]
