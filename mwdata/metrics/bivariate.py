from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cosine
from scipy.stats import iqr, spearmanr, f_oneway, levene
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import networkx as nx
import math
from mwdata.metrics.utils import hexbin
from mwdata import load_data
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import logging


class Scagnostics:
    """Scatter plot diagnostics

    Computes the scatter plot diagnostics ('scagnostics') for each pair of numeric features.

    The algorithm implemented here is described in Wilkinson, L., Anand, A., and Grossman, R. (2005),
     Graph-theoretic Scagnostics, in Proceedings of the IEEE Information Visualization 2005, pp. 157–164.

    Usage:
    ``

    df = pd.read_csv(path)

    Scagnostics(df).calculate()
    ``

    Args:
        data: A Pandas data frame
    """

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
            num_cols = data.shape[1]
        elif isinstance(data, pd.DataFrame):
            df = data.select_dtypes(['number'])
            self._colnames = df.columns.values
            self.data = df.to_numpy()
            num_cols = df.shape[1]
        else:
            df = load_data(data)
            df = df.select_dtypes(['number'])
            self._colnames = df.columns.values
            self.data = df.to_numpy()
            num_cols = df.shape[1]

        self.column_pairs = [(i, j) for i in range(num_cols) for j in range(num_cols) if j > i]

    @property
    def names(self):
        """list: column names"""
        return self._colnames

    def calculate(self, graphs=False, parallel=True):
        """ Compute scagnostics for all column pairs

        Args:
            graphs: If True, also return the network graphs and polygons calculated internally
            parallel: If True, run in parallel using maximum CPUs available

        Returns:
            A nested dictionary of metrics, keyed by the column pair tuple. For example, (i, j) refers to
            scagnostics for the column i and column j (zero-index).

            If `graphs` is False, each value is a dictionary of the 9 scagnostic measures
            If `graphs` is True, each value contains a dictionary with 2 keys: `Metrics` (for the scagnostics), and
            `Graphs` for the network graphs and polygons
        """
        if parallel:
            num_cores = cpu_count()
            p = Pool(num_cores)
            func = partial(Scagnostics.calculate_metrics, data=self.data, graphs=graphs)
            metrics = p.map(func, self.column_pairs)
            p.close()
            p.join()
            metrics = {k: v for d in metrics for k, v in d.items()}
        else:
            metrics = {}
            for pair in self.column_pairs:
                metrics[pair] = Scagnostics.calculate_metrics(pair, self.data, graphs)[pair]

        return metrics

    @staticmethod
    def calculate_metrics(pair, data=None, graphs=False):
        """ Calculate the 9 scagnostic measures for a single pair of columns

        The metrics are Outlier, Convex, Skinny, Skewed, Stringy, Straight, Monotonic, Clumpy, Striated

        Args:
            pair: The column pair, as a tuple of the column indices
            data: The data, as a numpy array
            graphs: If true, return all of the intermediate polygons & graphs used to calculate the measures

        Returns:
            A nested dictionary, keyed by the column pair tuple and then by metric
        """
        if data is None:
            raise ValueError("Data is required")

        array_data = Scagnostics.hex_bin(data[:, pair[0]], data[:, pair[1]])
        array_data = array_data[~np.isnan(array_data).any(axis=1)]

        try:
            array_data = StandardScaler().fit_transform(array_data)
            points = Scagnostics.convert_to_xy_tuples(array_data)

            triangulation = Delaunay(points)

            convex_hull = ConvexHull(points)

            minimum_spanning_tree = Scagnostics.minimum_spanning_tree(points, triangulation)

            alpha_hull = Scagnostics.concave_hull(points, minimum_spanning_tree, triangulation)

            results = {pair: {
                "Outlier": Scagnostics.outlying(minimum_spanning_tree),
                "Convex": Scagnostics.convex(alpha_hull, convex_hull),
                "Skinny": Scagnostics.skinny(alpha_hull),
                "Skewed": Scagnostics.skewed(minimum_spanning_tree),
                "Stringy": Scagnostics.stringy(minimum_spanning_tree),
                "Straight": Scagnostics.straight(minimum_spanning_tree, triangulation),
                "Monotonic": Scagnostics.monotonic(array_data),
                "Clumpy": Scagnostics.clumpy(minimum_spanning_tree),
                "Striated": Scagnostics.striated(minimum_spanning_tree, triangulation)}
            }

            if graphs:
                results[pair] = {"Metrics": results[pair],
                                 "Graphs": {
                                     "Triangulation": triangulation,
                                     "Convex Hull": convex_hull,
                                     "Minimum Spanning Tree": minimum_spanning_tree,
                                     "Alpha Hull": alpha_hull
                                 }}
        except Exception as e:
            warnings.warn("An error occurred during the scagnostic diagnostic computation. All metrics have been set to zero. {}".format(e))
            results = {pair: {
                "Outlier": 0.0,
                "Convex": 0.0,
                "Skinny": 0.0,
                "Skewed": 0.0,
                "Stringy": 0.0,
                "Straight": 0.0,
                "Monotonic": 0.0,
                "Clumpy": 0.0,
                "Striated": 0.0}
            }

        return results

    @staticmethod
    def hex_bin(x, y):
        """ Apply hexagon binning to the data

        Starts with a grid of 40*40 and halves the grid size until there are less than 250 non-empty bins

        Args:
            x: The x data
            y: The y data

        Returns:
            The binned hexagon center coordinates
        """
        counts, centers = [], []
        n_empty = 250
        grid = 80  # Starts at 40 x 40

        while n_empty >= 250:
            if grid == 1:
                break

            grid = int(grid / 2)

            counts, centers, hex_obj = hexbin(x, y, gridsize=(grid, grid))

            n_empty = (counts != 0).sum()

        nonzero = counts != 0
        centers = centers[nonzero]
        return centers

    @staticmethod
    def convert_to_xy_tuples(data):
        """ Convert arrays of x, y pairs in data into (x, y) tuples

        Args:
            data: A (n, 2) dimensional array. The 1st column is x-axis data and the 2nd column is y-axis data.

        Returns:
            A numpy array of (x, y) tuples
        """
        new_data = []
        for point in data:
            new_data.append((point[0], point[1]))

        return np.asarray(new_data)

    @staticmethod
    def squared_norm(v):
        """ Computes the squared norm (squared Euclidean distance) for a vector

        https://github.com/alpha-beta-soup/errorgeopy/blob/master/errorgeopy/utils.py

        Args:
            v: A vector

        Returns:
            The squared Euclidean distance
        """
        return np.linalg.norm(v) ** 2

    @staticmethod
    def circumcircle(points, simplex):
        """ Computes the circumcenter and circumradius of a triangle

        https://en.wikipedia.org/wiki/Circumscribed_circle#Circumcircle_equations
        https://github.com/alpha-beta-soup/errorgeopy/blob/master/errorgeopy/utils.py

        Args:
            points: The array of (x, y) points
            simplex: The triangles

        Returns:
            The circumcircle center and radius
        """
        A = [points[simplex[k]] for k in range(3)]
        M = np.asarray(
            [[1.0] * 4] +
            [[Scagnostics.squared_norm(A[k]), A[k][0], A[k][1], 1.0] for k in range(3)],
            dtype=np.float32)
        S = np.array([
            0.5 * np.linalg.det(M[1:, [0, 2, 3]]),
            -0.5 * np.linalg.det(M[1:, [0, 1, 3]])
        ])
        a = np.linalg.det(M[1:, 1:])
        b = np.linalg.det(M[1:, [0, 1, 2]])
        try:
            centre, radius = S / a, np.sqrt(b / a + Scagnostics.squared_norm(S) / a ** 2)
        except:
            logging.warning("Unexpected zero determinant in circumcircle calculation.")
            pass
        return centre, radius

    @staticmethod
    def get_alpha_complex(alpha, points, simplexes):
        """ Obtain the alpha shape

        https://github.com/alpha-beta-soup/errorgeopy/blob/master/errorgeopy/utils.py

        Args:
            alpha: The disk radius parameter for the alpha shape
            points: The array of (x, y) points
            simplexes: The triangles, i.e. from scipy.spatial.Delaunay

        Returns:
            A list of points in the alpha complex
        """
        return list(filter(lambda simplex:
                           Scagnostics.circumcircle(points, simplex)[1] < alpha,
                           simplexes))

    @staticmethod
    def concave_hull(points, minimum_spanning_tree, triangulation=None):
        """ Computes the concave hull (alpha shape) of a set of points.

        https://github.com/alpha-beta-soup/errorgeopy/blob/master/errorgeopy/utils.py

        Args:
            points: The array of (x, y) points
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object
            triangulation: A pre-computed Delaunay triangulation. Is recomputed from the points if not provided.

        Returns:
            The concave hull (alpha hull) Polygon
        """
        if triangulation is None:
            triangulation = Delaunay(np.array(points))

        weights = Scagnostics.get_mst_edges(minimum_spanning_tree)

        alpha_complex = []
        pct = 75
        while len(alpha_complex) == 0:
            omega = Scagnostics.calculate_omega(weights, pct)
            alpha_complex = Scagnostics.get_alpha_complex(omega, points, triangulation.simplices)
            pct = pct + 5

        x, y = [], []
        for s in alpha_complex:
            x.append([points[s[k]][0] for k in [0, 1, 2, 0]])
            y.append([points[s[k]][1] for k in [0, 1, 2, 0]])
        poly = Polygon(list(zip(x[0], y[0])))
        for i in range(1, len(x)):
            poly = poly.union(Polygon(list(zip(x[i], y[i]))))
        return poly

    @staticmethod
    def minimum_spanning_tree(points, triangulation=None):
        """ Determines the minimum spanning tree for the set of points

        Args:
            points: An array of (x, y) tuples
            triangulation: The Delaunay triangulation, i.e. from scipy.spatial.Delaunay

        Returns:
            The minimum spanning tree as a NetworkX graph object
        """
        if triangulation is None:
            triangulation = Delaunay(points)

        edges = set()

        for n in range(triangulation.nsimplex):
            # Edge a
            edge = sorted([triangulation.vertices[n, 0], triangulation.vertices[n, 1]])
            edges.add((edge[0], edge[1], np.linalg.norm(points[edge[0]] - points[edge[1]])))

            # Edge b
            edge = sorted([triangulation.vertices[n, 0], triangulation.vertices[n, 2]])
            edges.add((edge[0], edge[1], np.linalg.norm(points[edge[0]] - points[edge[1]])))

            # Edge c
            edge = sorted([triangulation.vertices[n, 1], triangulation.vertices[n, 2]])
            edges.add((edge[0], edge[1], np.linalg.norm(points[edge[0]] - points[edge[1]])))

        graph = nx.Graph()

        graph.add_weighted_edges_from(edges)

        mst = nx.minimum_spanning_tree(graph, algorithm='prim')

        return mst

    @staticmethod
    def get_mst_edges(mst, attr='weight'):
        """ Get all edges in the minimum spanning tree

        Args:
            mst: The minimum spanning tree, as a NetworkX graph object
            attr: The attribute name in the network graph to interpret as edge weight

        Returns:
            A list of edge lengths (weights)
        """
        return [x[2][attr] for x in mst.edges(data=True)]

    @staticmethod
    def calculate_omega(weights, pct=75):
        """ Calculate the omega value to determine alpha shape

        Args:
            weights: The edge lengths (weights) from the minimum spanning tree
            pct: The percentile cutoff for outliers

        Returns:
            Omega
        """
        return np.percentile(weights, pct) + 1.5 * iqr(weights)

    @staticmethod
    def longest_shortest_path(mst, where=False):
        """ Calculate the longest shortest-path in the minimum spanning tree

        Each node in the minimum spanning tree has a shortest path to each other node. Compute the longest path
        out of all shortest paths in the spanning tree. This represents the 'diameter' of the minimum spanning tree.

        Args:
            mst: The minimum spanning tree, as a NetworkX graph object
            where: If True, return the start and end nodes instead of the path length

        Returns:
            If where is False, returns the path length for the longest shortest path
            If where is True, returns the start node and end node of the longest shortest path
        """
        all_shortest_paths = nx.all_pairs_dijkstra_path_length(mst, weight='weight')
        nodes = [node for node in all_shortest_paths]
        all_paths = {(n[0], k): v for n in nodes for k, v in n[1].items()}
        if where:
            return max(all_paths, key=all_paths.get)
        else:
            return max(all_paths.values())

    @staticmethod
    def outlying(minimum_spanning_tree):
        """ Outlier metric

        Args:
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object

        Returns:
            Metric value
        """
        vertices = [v[0] for v in list(minimum_spanning_tree.degree()) if v[1] == 1]
        edges = [list(minimum_spanning_tree.edges(v, data=True)) for v in vertices]
        outer_weights = [x[0][2]['weight'] for x in edges]

        weights = Scagnostics.get_mst_edges(minimum_spanning_tree)
        omega = Scagnostics.calculate_omega(weights)

        out_edges = [w for w in outer_weights if w > omega]
        return np.sum(out_edges) / np.sum(outer_weights)

    @staticmethod
    def convex(alpha_hull, convex_hull):
        """ Convex metric

        Args:
            alpha_hull: The alpha hull shape, as a Shapely polygon
            convex_hull: The convex hull shape

        Returns:
            Metric value
        """
        return alpha_hull.area / convex_hull.volume  # Scipy ConvexHull Volume = Area in 2D

    @staticmethod
    def skinny(alpha_hull):
        """ Skinny metric

        Args:
            alpha_hull: The alpha hull shape, as a Shapely polygon

        Returns:
            Metric value
        """
        return 1 - math.sqrt(4 * math.pi * alpha_hull.area) / alpha_hull.length

    @staticmethod
    def stringy(minimum_spanning_tree):
        """ Stringy metric

        Args:
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object

        Returns:
            Metric value
        """
        mst_diam = Scagnostics.longest_shortest_path(minimum_spanning_tree)
        mst_length = minimum_spanning_tree.size(weight='weight')
        return mst_diam / mst_length

    @staticmethod
    def straight(minimum_spanning_tree, triangulation):
        """ Straight metric

        Args:
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object
            triangulation: The Delaunay triangulation, i.e. from scipy.spatial.Delaunay

        Returns:
            Metric value
        """
        mst_diam = Scagnostics.longest_shortest_path(minimum_spanning_tree)
        points_euc = Scagnostics.longest_shortest_path(minimum_spanning_tree, True)
        dist_euc = np.linalg.norm(triangulation.points[points_euc[0]] -
                                  triangulation.points[points_euc[1]])
        return dist_euc / mst_diam

    @staticmethod
    def monotonic(array_data):
        """ Monotonic metric

        Args:
            array_data: The (n,2) data array

        Returns:
            Metric value
        """
        return spearmanr(array_data[:, 0], array_data[:, 1])[0] ** 2

    @staticmethod
    def skewed(minimum_spanning_tree):
        """ Skewed metric

        Args:
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object

        Returns:
            Metric value
        """
        weights = Scagnostics.get_mst_edges(minimum_spanning_tree)
        return (max(0, np.percentile(weights, 90) - np.percentile(weights, 50))) / \
               (abs(np.percentile(weights, 90) - np.percentile(weights, 10)))

    @staticmethod
    def clumpy(minimum_spanning_tree):
        """ Clumpy metric

        Args:
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object

        Returns:
            Metric value
        """
        all_edges = minimum_spanning_tree.edges(data=True)
        max_clump = 0
        for e in all_edges:
            n1 = e[0]
            n2 = e[1]
            new_graph = minimum_spanning_tree.copy()
            min_length = e[2]['weight']
            edges_to_break = [(e[0], e[1]) for e in all_edges if e[2]['weight'] >= min_length]
            new_graph.remove_edges_from(edges_to_break)

            sub_graph_1_nodes = nx.node_connected_component(new_graph, n1)
            sub_graph_2_nodes = nx.node_connected_component(new_graph, n2)

            sub_graph_1_edges = [x[2]['weight'] for x in new_graph.edges(sub_graph_1_nodes, data=True)]
            sub_graph_2_edges = [x[2]['weight'] for x in new_graph.edges(sub_graph_2_nodes, data=True)]

            if len(list(sub_graph_1_edges)) <= len(list(sub_graph_2_edges)):
                if len(list(sub_graph_1_edges)) > 0:
                    max_edge = max(list(sub_graph_1_edges))
                else:
                    continue
            else:
                if len(list(sub_graph_2_edges)) > 0:
                    max_edge = max(list(sub_graph_2_edges))
                else:
                    continue

            clump_j = 1 - max_edge / min_length
            if clump_j > max_clump:
                max_clump = clump_j
        return max_clump

    @staticmethod
    def striated(minimum_spanning_tree, triangulation):
        """ Striated metric

        Args:
            minimum_spanning_tree: The minimum spanning tree, as a NetworkX graph object
            triangulation: The Delaunay triangulation, i.e. from scipy.spatial.Delaunay

        Returns:
            Metric value
        """
        vertices = [v[0] for v in list(minimum_spanning_tree.degree()) if v[1] == 2]
        point_set = [list(minimum_spanning_tree.edges(v)) for v in vertices]
        tri_points = triangulation.points

        vector_pairs = [[(tri_points[edge[0]] - tri_points[edge[1]]) for edge in edge_pair] for edge_pair in point_set]

        angles = [abs(1 - cosine(vector[0], vector[1])) for vector in vector_pairs]

        # sum(angles) / len(angles)
        return len([a for a in angles if a > 0.70]) / len(angles)


def varying(grp, alpha=0.01):
    """ Identifies varying box plots (i.e. with different means) using the one-way ANOVA test

    Args:
        grp: The groups from `split_by_category`
        alpha: The significance level

    Returns:
        True if statistically significant
    """
    F, p = f_oneway(*grp)
    return p <= alpha


def heteroscedastic(grp, alpha=0.01):
    """ Identifies heteroscedasticity in box plots using the Brown-Forscythe test

    Args:
        grp: The groups from `split_by_category`
        alpha: The significance level

    Returns:
        True if statistically significant
    """
    W, p = levene(*grp, center='median')
    return p <= alpha
