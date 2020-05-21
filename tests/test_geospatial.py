import mwdata.geospatial.mapping as mw
import pytest


@pytest.fixture
def data_loader():
    counties = mw.load_data("data/geo/tl_2018_us_county.shp")
    pop = mw.load_data(
        "data/geo/DEC_10_SF1_GCTPH1.US05PR_with_ann.csv",
        kwargs={"encoding": "latin1", "skiprows": 1, "dtype": {"Target Geo Id2": str}},
    )
    geo = counties.merge(pop, left_on="GEOID", right_on="Target Geo Id2")
    geo["state"] = geo["Geographic area"].map(lambda x: x.split("-")[1].strip())
    data = geo[~geo["state"].isin(["Hawaii", "Alaska"])]
    return data.sample(n=50, replace=True, random_state=1)


def test_maps_kde(data_loader):
    fig = mw.maps(data=data_loader, map_type="kde")
    assert fig is not None
    fig_2 = mw.maps(data=data_loader, map_type="kde", kde_kwargs={"cbar": False})
    assert fig_2 is not None


def test_maps_choropleth(data_loader):
    fig = mw.maps(data=data_loader, color="ALAND")
    assert fig is not None
    fig_2 = mw.maps(data=data_loader)
    assert fig_2 is not None


def test_implementation_error():
    with pytest.raises(NotImplementedError):
        mw.maps("data/er_data.csv")


def test_value_errors():
    with pytest.raises(ValueError):
        mw.maps("data/geo/tl_2018_us_county.shp", color="NAME")
    with pytest.raises(ValueError):
        mw.maps("data/geo/tl_2018_us_county.shp", map_type="Random plot")
