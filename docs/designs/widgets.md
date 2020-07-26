# Design Proposal for Widgets

## Motivation

Feature in data-describe may be consolidated into "widgets", a logical grouping of functionality usually associated with a single visualization as its primary output e.g. cluster analysis, data heatmap, etc. These widget visualizations may be accompanied by additional secondary visualizations (for diagnostics). Additionally, end users may want to access internally computed objects to modify or extend the analysis.

## Goals

- Consolidate multiple outputs of each "widget" into a single object
- Define an interface (design pattern) to allow for returning secondary visualizations and other objects, such as the estimators used internally
- (UX) Allow the primary output (usually a visualization) to be shown by default i.e. when the object is the last line in a Jupyter Notebook cell

## Non-Goals

- Design or provide methods for re-running (computing) the feature with changed inputs; this design document only defines the widget as a container for shared outputs and not as a reusable object

## UI or API

An abstract class has been added to `widgets.py`:
```python
class BaseWidget(ABC):
    """Base widget."""

    def __init__(self, viz_backend=None, **kwargs):
        self.viz_backend = viz_backend
        """Expected attributes."""
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __repr__(self):
        return "data-describe Base Widget"

    def _repr_html_(self):
        return display(self.show())

    @abstractmethod
    def show(self, viz_backend=None):
        """Show the default output."""
        raise NotImplementedError("No default visualization or output has been defined for this widget.")
```


## Design

Each feature should subclass the `BaseWidget`:

### `__init__`
Arguments in the `__init__()` signature should primarily be used to indicate the attributes that are expected on the widget. Not all expected attributes must be assigned during initialization; they may be assigned after instantiation.

> One can call `super(BaseWidget, self).__init__(**kwargs)` in the subclass to self-assign keyword arguments.

### `__repr__()`
`__repr__` should be implemented and return a human-readable string that provides a short description of the widget, e.g. "data-describe Cluster Widget"

### `_repr_html_()`
`_repr_html_` is used to [display the object](https://ipython.readthedocs.io/en/stable/config/integrating.html) in a IPython/Jupyter setting when the object is the last line in a cell. It expects output from the `show()` method.

### `show()`
The `show` method should be implemented and return the default output for this widget, e.g. a Matplotlib Axes object, Plotly Figure, or Pandas DataFrame.

In most cases, the implementation should check `self.viz_backend` that was optionally passed by the end user to the top level function.

### Additional Attributes and Methods
This design does not explicitly set any guidelines for other attributes or methods. However, any attributes or methods that are expected to be implemented by or utilized by backend implementations are strongly recommended to be defined in the widget class definition (so that they are documented).

## Alternatives Considered

### Specify outputs using parameters
Each feature could take an additional parameter `return_value` to specify the desired output. 

```
def widget(..., return_value="plot"):
    if return_value == "plot":
        return plot
    elif return_value == "data":
...
```

*Disadvantages:*

- Adds clutter to the function signature
- Requires additional logic in the function to parse `return_value`
- Users only receive one output and cannot get other desired outputs without re-running the function

### Return a tuple of multiple outputs
```
return obj1, obj2
```

*Disadvantages:*

- API does not scale well with new or changing outputs
- Cannot return a "default" output that displays in notebooks without unpacking the tuple

## Roadmap
Future enhancements for consideration include:
- Implement methods for re-computation of new inputs
- Integrate or connect widget with data-describe Jupyter Lab Extension
- Fully summarize the contents of the widget for reporting