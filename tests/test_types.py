# import datetime
# import decimal
# import six

# import pandas as pd
# import pytest

# from data_describe.type.autotype import (
#     guess_dtypes,
#     guess_series_dtypes,
#     meta_features,
#     get_class_instance_by_name,
#     cast_dtypes,
#     select_dtypes,
# )
# from data_describe.type.dtypes import (
#     BaseType,
#     StringType,
#     IntegerType,
#     DecimalType,
#     CategoryType,
#     DateTimeType,
#     ReferenceType,
#     BoolType,
#     NumericType,
# )


# @pytest.fixture
# def load_data():
#     df = pd.DataFrame(
#         {
#             "glimepiride.pioglitazone": [True, False],
#             "number_outpatient": [1, 2],
#             "diag_2_desc": ["is a medicine", "is not a medicine"],
#             "nateglinide": ["a", "b"],
#             "readmitted": ["true", "false"],
#         }
#     )
#     return df


# def test_guess_types(load_data):
#     assert guess_dtypes(load_data) == {
#         "glimepiride.pioglitazone": "Boolean",
#         "number_outpatient": "Integer",
#         "diag_2_desc": "String",
#         "nateglinide": "Category",
#         "readmitted": "Boolean",
#     }
#     with pytest.raises(ValueError):
#         assert guess_dtypes([1])


# def test_guess_series_base_types(load_data):
#     assert guess_series_dtypes(load_data["diag_2_desc"]) == "String"
#     assert guess_series_dtypes(load_data["diag_2_desc"], sample_size=0.2) == "String"
#     assert guess_series_dtypes(load_data["diag_2_desc"].iloc[5:15]) == "String"


# def guess_series_type(load_data):
#     assert guess_series_type(load_data["diag_2_desc"].iloc[5:15]) == {
#         "diag_2_desc": "String"
#     }


# def test_select_dtypes(load_data):
#     assert select_dtypes(
#         load_data, types=["String"], dtypes={"diag_2_desc": "String"}
#     ).columns == ["diag_2_desc"]


# def test_cast_types(load_data):
#     df = cast_dtypes(
#         load_data,
#         dtypes={"readmitted": "Category"},
#         exclude=["glimepiride.pioglitazone"],
#     )
#     assert (
#         str(df.dtypes.values)
#         == "[dtype('O') Int64Dtype() dtype('O') dtype('O') dtype('O')]"
#     )
#     with pytest.raises(ValueError):
#         assert cast_dtypes(load_data, dtypes={"readmitted": "NotAType"})


# def test_get_instances():
#     assert isinstance(get_class_instance_by_name("String"), object)


# def test_meta(load_data):
#     df = load_data
#     assert isinstance(meta_features(df.iloc[:, 0]), dict)


# def test_basetype():
#     b = BaseType(null_values=["N.A."])
#     assert isinstance(b, BaseType)
#     assert isinstance(b.null_values, list)
#     assert b.null_values == ["N.A."]
#     assert b.weight == 0
#     assert isinstance(b.result_type, tuple)
#     assert b.name == "None"
#     assert b.test(1) == 1
#     assert b.test(None) == 1
#     assert not b.test_meta({})
#     assert b.cast(1) == 1
#     b.null_values = ["NAAA"]
#     assert b.null_values == ["NAAA"]
#     b.weight = 1.5
#     assert b.weight == 1.5


# def test_stringtype():
#     s = StringType()
#     assert isinstance(s, BaseType)
#     assert isinstance(s, StringType)
#     assert s.cast(1) == "1"
#     assert s.cast("ab") == "ab"
#     assert s.cast("na") is None
#     assert s.test("categoryA") == 0
#     assert s.test("this is a long string?") == 1
#     assert s.weight == 2
#     assert s.result_type == six.string_types
#     assert s.name == "String"


# def test_categorytype():
#     c = CategoryType()
#     assert isinstance(c, BaseType)
#     assert isinstance(c, StringType)
#     assert isinstance(c, CategoryType)
#     assert c.test("categoryA") == 1
#     assert c.test("this is a string?") == 1
#     assert c.weight == 1
#     assert c.name == "Category"


# def test_integertype():
#     i = IntegerType()
#     assert isinstance(i, BaseType)
#     assert isinstance(i, NumericType)
#     assert isinstance(i, IntegerType)
#     assert i.cast(1.0) == 1
#     with pytest.raises(ValueError):
#         assert i.cast(1.1)
#     assert i.cast("3") == 3
#     assert i.cast("na") is None
#     assert i.test("categoryA") == -1
#     assert i.test("46") == 1
#     assert i.test(25) == 1
#     assert i.test(2.652) == -1
#     assert i.test(8.0000) == 0
#     assert i.weight == 6
#     assert i.result_type == (pd.Int64Dtype,) + six.integer_types
#     assert i.name == "Integer"


# def test_decimaltype():
#     d = DecimalType()
#     assert isinstance(d, BaseType)
#     assert isinstance(d, NumericType)
#     assert isinstance(d, DecimalType)
#     assert d.cast(1.0) == 1.0
#     with pytest.raises(ValueError):
#         assert d.cast("abc")
#     assert d.cast("3.0") == 3.0
#     assert d.cast("na") is None
#     assert d.cast(4) == 4.0
#     assert d.test("categoryA") == -1
#     assert d.test("46") == 0
#     assert d.test(25) == 0
#     assert d.test(2.652) == 1
#     assert d.test(8.0000) == 1
#     assert d.weight == 4
#     assert d.result_type == (float, decimal.Decimal)
#     assert d.name == "Decimal"


# def test_booltype():
#     b = BoolType()
#     assert isinstance(b, BaseType)
#     assert isinstance(b, BoolType)
#     assert b.weight == 7
#     assert b.result_type == (bool,)
#     assert b.name == "Boolean"
#     assert b.true_values == ["yes", "true", "0", "y", "t"]
#     assert b.false_values == ["no", "false", "1", "n", "f"]
#     b.true_values = ["yea"]
#     assert b.true_values == ["yea"]
#     assert b.test("categoryA") == -1
#     assert b.test(1) == 1
#     assert b.test(False) == 1
#     assert b.test("yea") == 1


# def test_datetimetype():
#     d = DateTimeType()
#     assert isinstance(d, BaseType)
#     assert isinstance(d, DateTimeType)
#     assert d.weight == 3
#     assert isinstance(d.result_type, tuple)
#     assert d.name == "DateTime"
#     assert d.format is None
#     assert d.test("2019-92-44") == -1
#     assert d.test("2019-01-23") == 1
#     assert d.test("Monday, the first") == -1
#     assert d.test("January 1, 1938") == 1
#     assert isinstance(d.cast("2019-04-24"), datetime.datetime)
#     assert d.cast("na") is None
#     d.format = "%Y-%d"
#     assert d.test("01 2045") == -1
#     assert d.test("2045-01") == 1
#     assert isinstance(d.cast("2045-01"), datetime.datetime)


# def test_referencetype():
#     r = ReferenceType()
#     assert r.weight == 0
#     assert isinstance(r.result_type, tuple)
#     assert r.name == "Reference"

#     assert r.test_meta({"size": 22, "cardinality": 22}) == 1
#     assert r.test_meta({"size": 21, "cardinality": 22}) == -1
#     with pytest.raises(ValueError):
#         assert r.test_meta({"size": 15})