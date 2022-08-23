from typing import Set
from collections import Iterable


class ConditionalIndependence:
    """Store and print conditional independence relations in the form X⫫Y|{Z1, Z2, ..., Zn}."""

    def __init__(self, X: str, Y: str, Z: Set[str] = None):
        self.X = X
        self.Y = Y
        self.Z = Z if Z is not None else []

    def __str__(self):
        base_str = f"{self.X}⫫{self.Y}"
        if self.Z:
            zs_list = list(self.Z)
            zs_list.sort()
            adjustment_set_str = "{"
            for z in zs_list:
                adjustment_set_str += f"{z}, "
            adjustment_set_str = adjustment_set_str[:-2] + "}"
            base_str += f"|{adjustment_set_str}"
        return base_str

    def __repr__(self):
        return str(self)

    def __eq__(self, other):

        # Check if X is the same
        if self.X != other.X:
            return False

        # Check if Y is the same
        if self.Y != other.Y:
            return False

        # Check if Z is the same
        if self.Z != other.Z:
            return False
        return True
