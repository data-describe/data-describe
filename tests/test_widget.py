import sys
from unittest import mock

import pytest
import IPython

from data_describe._widget import BaseWidget


@pytest.fixture
@mock.patch.multiple(BaseWidget, __abstractmethods__=set())
def basewidget():
    return BaseWidget(custom_arg=1)


def test_str(basewidget):
    assert str(basewidget) == "data-describe Base Widget", "__str__ had incorrect value"


def test_repr_html(basewidget):
    with mock.patch.object(IPython, "display") as mock_display:
        with pytest.raises(NotImplementedError):
            basewidget._repr_html_()
            mock_display.assert_called_with()


def test_repr_html_no_ipython(basewidget, monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)
    with pytest.raises(NotImplementedError):
        basewidget._repr_html_()
