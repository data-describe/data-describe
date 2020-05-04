import six
import decimal
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import date, time, datetime, timedelta
from dateutil.parser import parse, isoparse
import locale
import sys


class BaseType(object):
    """Base data type, unidentified"""

    def __init__(self, weight=0, name="None", result_type=None, null_values=None):
        """A class to test for data types

        Args:
            weight: The relative weight for this data type
            name: The name
            result_type: A tuple of Python types that the data can take, including primitive types, numpy, or Pandas types
            null_values: A list of strings which will be evaluated to mean missing values
        """
        self._weight = weight

        self._name = name

        if not result_type:
            self._result_type = (type(None),)
        else:
            self._result_type = result_type

        if null_values:
            self.null_values = null_values
        else:
            self._null_values = ['', 'na', 'n/a', 'null']

    def test(self, value):
        """Test if a given value passes the data type validation rules

        Args:
            value: The value to be tested

        Returns:
            A weighted "score" which may be in {-1, 0, 1}
        """
        if isinstance(value, self.result_type):
            return True
        else:
            try:
                if str(self.cast(value)) == str(value):
                    return 1
                else:
                    return 0
            except:
                return -1

    @staticmethod
    def test_meta(meta):
        """Given some metadata about the entire feature / column, test if it passes the data type validation rules

        Args:
            meta: A dictionary containing meta features

        Returns:
            A weighted "score" which may be in {-1, 0, 1}
        """
        return False

    def cast(self, value):
        """Cast the value to this data type

        Args:
            value: The value to be converted

        Returns:
            The data value casted to the current type
        """
        return value

    @classmethod
    def instances(cls):
        """Return all instances of this type class

        Returns:
            A list of instances
        """
        return [cls()]

    @property
    def null_values(self):
        """Get the null value strings

        Returns:
            A list of null value strings
        """
        return self._null_values

    @null_values.setter
    def null_values(self, value):
        """Set the null value strings

        Args:
            value: A list of strings to be evaluated as null values (ignored)
        """
        self._null_values = value

    def append_null_value(self, value):
        """Append a string value to be evaluated as a null value

        Args:
            value: A string for which exact matches will be counted as nulls

        Returns:

        """
        self._null_values.append(value)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def result_type(self):
        return self._result_type

    @property
    def name(self):
        return self._name


class StringType(BaseType):
    """String data type"""

    def __init__(self, weight=2, name="String", result_type=six.string_types, null_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)

    def cast(self, value):
        if str(value).strip().lower() in self.null_values:
            return None
        if isinstance(value, self.result_type):
            return value
        else:
            return str(value)

    def test(self, value):
        if str(value).strip().find(" ") > -1:
            return 1
        else:
            return 0


class CategoryType(StringType):
    def __init__(self, weight=1, name="Category", result_type=None, null_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)

    def test(self, value):
        return 1


class NumericType(BaseType):
    """Numeric type for selection"""

    def __init__(self, weight=0, name="Numeric", result_type=None, null_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)


class IntegerType(NumericType):
    """Integer data type"""

    def __init__(self, weight=6, name="Integer", result_type=(pd.Int64Dtype,) + six.integer_types, null_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)

    def test(self, value):
        if str(value)[0] == '0' and len(str(value)) > 1:
            return -1  # No leading zeros
        try:
            if isinstance(locale.atoi(str(value)), self.result_type):
                if locale.localeconv()['decimal_point'] in str(value):
                    return -1  # No decimal points
                else:
                    return 1  # Convertible to integer
            else:
                return -1  # Not convertible to integer
        except:
            return super().test(value)

    def cast(self, value):
        if str(value).strip().lower() in self.null_values:
            return None

        try:
            value = float(value)
        except:
            return locale.atoi(value)

        if value.is_integer():
            return int(value)
        else:
            raise ValueError('Invalid integer: {}'.format(value))


class DecimalType(NumericType):
    """Decimal or float data type"""

    def __init__(self, weight=4, name="Decimal", result_type=(float, decimal.Decimal), null_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)

    def test(self, value):
        try:
            if isinstance(locale.atof(str(value)), self.result_type):
                if locale.localeconv()['decimal_point'] in str(value):
                    return 1  # Has decimal point
                else:
                    return 0  # Convertible but no decimal point
            else:
                return -1  # Not convertible
        except:
            return super().test(value)

    def cast(self, value):
        if str(value).strip().lower() in self.null_values:
            return None
        try:
            return decimal.Decimal(value)
        except:
            value = locale.atof(value)
            if sys.version_info < (2, 7):
                value = str(value)
            return decimal.Decimal(value)


class BoolType(BaseType):
    """Boolean data type"""

    def __init__(self, weight=7, name="Boolean", result_type=(bool,), null_values=None, true_values=None,
                 false_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)

        if true_values:
            self._true_values = true_values
        else:
            self._true_values = ['yes', 'true', '0', 'y', 't']

        if false_values:
            self.false_values = false_values
        else:
            self._false_values = ['no', 'false', '1', 'n', 'f']

    def test(self, value):
        s = str(value).strip().lower()
        if s in self.null_values:
            return 0
        if s in self.true_values:
            return 1
        if s in self.false_values:
            return 1

        return super().test(value)

    def cast(self, value):
        s = str(value).strip().lower()
        if s in self.null_values:
            return None
        if s in self.true_values:
            return True
        if s in self.false_values:
            return False
        raise ValueError('Not a recognized boolean type: {}'.format(value))

    @property
    def true_values(self):
        return self._true_values

    @true_values.setter
    def true_values(self, value):
        self._true_values = value

    @property
    def false_values(self):
        return self._false_values

    @false_values.setter
    def false_values(self, value):
        self._false_values = value


class DateTimeType(BaseType):
    """Date/time data type"""

    def __init__(self, weight=3, name="DateTime", result_type=(np.datetime64, datetime, date, time, timedelta),
                 null_values=None, date_format=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)
        self._format = date_format

    def test(self, value):
        if self._format is not None:
            try:
                dt.strptime(value, self._format)
                return 1
            except:
                return super().test(value)
        else:
            try:
                isoparse(value)
                return 1
            except ValueError:
                try:
                    parsed_dt = parse(value)
                    if (str(parsed_dt.year) in str(value) and
                            str(parsed_dt.month) in str(value) and
                            str(parsed_dt.day) in str(value)):
                        return 1
                    else:
                        return 0
                except:
                    return super().test(value)
            except:
                return super().test(value)

    def cast(self, value):
        if isinstance(value, self.result_type):
            return value
        elif str(value).strip().lower() in self.null_values:
            return None
        elif self._format is None:
            return parse(str(value))
        else:
            return dt.strptime(value, self._format)

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):
        self._format = value


class ReferenceType(BaseType):
    """Reference/ID data type"""

    def __init__(self, weight=0, name="Reference", result_type=None, null_values=None):
        super().__init__(weight=weight, name=name, result_type=result_type, null_values=null_values)

    @staticmethod
    def test_meta(meta):
        if 'size' in meta.keys() and 'cardinality' in meta.keys():
            if meta['size'] == meta['cardinality']:
                return 1
            else:
                return -1
        else:
            raise ValueError("The meta features dictionary requires the number of records ('size') "
                             "and number of unique records ('cardinality') to test for the ReferenceType.")


TYPES = [CategoryType, StringType, DecimalType, IntegerType, DateTimeType, BoolType, ReferenceType]
