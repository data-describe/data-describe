import data_describe.core.distribution as ddd
from data_describe.metrics.univariate import spikey, skewed


def compute_distribution(
    data,
    diagnostic=True,
    max_categories=20,
    label_name="(OTHER)",
    spike_factor=10,
    skew_factor=3,
    **kwargs
):
    """Compute distribution metrics.

    Args:
        data (DataFrame): The data
        diagnostic (bool, optional): If True, will compute diagnostics used to select "interesting" plots.
        max_categories (int, optional): Maximum categories to display. Defaults to 20.
        label_name (str, optional): The label to use for categories combined after max_categories. Defaults to "(OTHER)".
        spike_factor (int, optional): The spikey-ness factor used to flag spikey histograms. Defaults to 10.
        skew_factor (int, optional): The skew-ness factor used to flag skewed histograms. Defaults to 3.

    Returns:
        DistributionWidget
    """
    num_data = data.select_dtypes("number")
    cat_data = data.select_dtypes(
        exclude=["number", "datetime", "timedelta", "datetimez"]
    )

    spikeyness = num_data.apply(spikey, axis=1) if diagnostic else None
    skewness = num_data.apply(skewed, axis=1) if diagnostic else None
    cardinality = cat_data.nunique() if diagnostic else None
    categories_to_squash = cardinality[cardinality > max_categories].index if diagnostic else None

    return ddd.DistributionWidget(
        input_data=data,
        num_data=num_data,
        cat_data=cat_data,
        spikeyness=spikeyness,
        skewness=skewness,
        max_categories=max_categories,
        label_name=label_name,
        spike_factor=spike_factor,
        skew_factor=skew_factor,
        categories_to_squash=categories_to_squash,
    )
