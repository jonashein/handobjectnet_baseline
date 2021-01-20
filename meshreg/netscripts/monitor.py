import numpy as np
import pickle
import os
import math
from collections.abc import Iterable
import torch
from matplotlib import pyplot as plt
import tikzplotlib
import plotly.offline as py
import plotly.subplots as pyplots
import plotly.graph_objs as go

# Seaborn bright color palette
COLORS=[(0.00784313725490196, 0.24313725490196078, 1.0),
        (1.0, 0.48627450980392156, 0.0),
        (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
        (0.9098039215686274, 0.0, 0.043137254901960784),
        (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
        (0.6235294117647059, 0.2823529411764706, 0.0),
        (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
        (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
        (1.0, 0.7686274509803922, 0.0),
        (0.0, 0.8431372549019608, 1.0)]

class MetricMonitor:
    """TODO"""

    def __init__(self, filepath=None):
        """TODO"""
        # load data from filepath
        # data layout: {metric: {split: {epoch: ...}}}
        self.metrics = {}
        self.data = {}
        if filepath is not None:
            assert os.path.isfile(filepath), "File not found: {}".format(filepath)
            self.data = pickle.load(open(filepath, "rb"))
        self.add_meta("obj_verts_2d", {"axis_label": "Threshold (pixel)",
                                       "axis_limits": (0.0, 100.0)})
        self.add_meta("obj_keypoints_2d", {"axis_label": "Threshold (pixel)",
                                           "axis_limits": (0.0, 100.0)})
        self.add_meta("obj_verts_3d", {"axis_label": "Threshold (mm)",
                                       "axis_scale": 1000.0, # convert m to mm
                                       "axis_limits": (0.0, 100.0)})
        self.add_meta("obj_depth_3d", {"axis_label": "Threshold (mm)",
                                       "axis_scale": 1000.0,  # convert m to mm
                                       "axis_limits": (0.0, 100.0)})
        self.add_meta("obj_translation_3d", {"axis_label": "Threshold (mm)",
                                             "axis_scale": 1000.0,  # convert m to mm
                                             "axis_limits": (0.0, 100.0)})
        self.add_meta("obj_rotation_3d", {"axis_label": "Threshold (deg)",
                                          "axis_limits": (0.0, 50.0)})
        self.add_meta("obj_drilltip_trans_3d", {"axis_label": "Threshold (mm)",
                                                "axis_scale": 1000.0,  # convert m to mm
                                                "axis_limits": (0.0, 100.0)})
        self.add_meta("obj_drilltip_rot_3d", {"axis_label": "Error in deg",
                                              "axis_limits": (0.0, 50.0)})
        self.add_meta("hand_joints_3d", {"axis_label": "Threshold (mm)",
                                         "axis_scale": 1000.0,  # convert m to mm
                                         "axis_limits": (0.0, 50.0)})
        self.add_meta("hand_joints_2d", {"axis_label": "Threshold (pixel)",
                                         "axis_limits": (0.0, 50.0)})


    def add(self, split, epoch, batch):
        """TODO"""
        assert batch is not None and isinstance(batch,dict), "invalid batch!"

        for m, values in batch.items():
            if m not in self.data.keys():
                self.data[m] = {}
            if split  not in self.data[m].keys():
                self.data[m][split] = {}
            if epoch not in self.data[m][split].keys():
                self.data[m][split][epoch] = []
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self.data[m][split][epoch].append(values)


    def add_meta(self, metric, meta_dict):
        """Add metadata to metric"""
        self.metrics[metric] = meta_dict


    def get(self, metrics=None, splits=None, epochs=None, squeeze=True):
        """TODO"""
        # parse arguments
        if metrics is not None and not isinstance(metrics, list):
            metrics = [metrics]
        if splits is not None and not isinstance(splits, list):
            splits = [splits]
        if epochs is not None and not isinstance(epochs, list):
            epochs = [epochs]

        # get requested metrics,splits,epochs
        if metrics is None:
            metrics = self.data.keys()
        if splits is None:
            splits = []
            for m, s_dict in self.data.items():
                splits.extend([s for s in s_dict.keys() if s not in splits])

        results = {}
        for m in metrics:
            if m not in self.data.keys():
                continue
            results[m] = {}
            for s in splits:
                if s not in self.data[m].keys():
                    continue
                results[m][s] = {}

                if epochs is None:
                    ms_epochs = self.data[m][s].keys()
                else:
                    ms_epochs = epochs
                for e in ms_epochs:
                    if e not in self.data[m][s].keys():
                        continue
                    try:
                        res = [np.expand_dims(it, axis=0) if it.ndim == 0 else it for it in self.data[m][s][e]]
                        results[m][s][e] = np.concatenate(res, axis=0)
                    except Exception as e:
                        print("Metric: {}".format(m))
                        print("Split: {}".format(s))
                        print("Epoch: {}".format(e))
                        print(e)


        # squeeze dicts with a single key if enforced by given arguments
        if squeeze:
            for m in results.keys():
                for s in results[m].keys():
                    for e in results[m][s].keys():
                        val = np.array(results[m][s][e])
                        if (val.size == 1):
                            results[m][s][e] = val.item()
                    if epochs is not None and len(epochs) == 1:
                        assert len(results[m][s].keys()) == 1
                        _, results[m][s] = list(results[m][s].items())[0]
                if splits is not None and len(splits) == 1:
                    assert len(results[m].keys()) == 1
                    _, results[m] = list(results[m].items())[0]
            if metrics is not None and len(metrics) == 1:
                assert len(results.keys()) == 1
                _, results = list(results.items())[0]

        return results


    def means_per_epoch(self, metrics=None, splits=None, epochs=None, squeeze=False, ignore_nan=False):
        """TODO"""
        result = self.get(metrics, splits, epochs, squeeze=squeeze)
        for m, s_dict in result.items():
            for s, e_dict in s_dict.items():
                for e, val in e_dict.items():
                    if ignore_nan:
                        val = val[np.isfinite(val)]
                    result[m][s][e] = np.mean(val)
        return result

    def std_per_epoch(self, metrics=None, splits=None, epochs=None, squeeze=False, ignore_nan=False):
        """TODO"""
        result = self.get(metrics, splits, epochs, squeeze=squeeze)
        for m, s_dict in result.items():
            for s, e_dict in s_dict.items():
                for e, val in e_dict.items():
                    if ignore_nan:
                        val = val[np.isfinite(val)]
                    result[m][s][e] = np.std(val)
        return result


    def percentile_per_epoch(self, percentile, metrics=None, splits=None, epochs=None, squeeze=False, ignore_nan=False):
        """TODO"""
        result = self.get(metrics, splits, epochs, squeeze=squeeze)
        for m, s_dict in result.items():
            for s, e_dict in s_dict.items():
                for e, val in e_dict.items():
                    if ignore_nan:
                        val = val[np.isfinite(val)]
                    result[m][s][e] = np.percentile(val, percentile)
        return result


    def save_metrics(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.data, f)


    def update(self, other, self_prefix=None, other_prefix=None):
        # updates itself with the values from the other monitor instance
        assert other is not None and isinstance(other, MetricMonitor), "Invalid argument: other"

        if self_prefix is not None:
            self.data = _add_prefix_to_splits(self.data, self_prefix)
        other_data = other.data
        if other_prefix is not None:
            other_data = _add_prefix_to_splits(other_data, other_prefix)

        for m in other_data.keys():
            if m not in self.data.keys():
                self.data[m] = other_data[m]
                continue
            for s in other_data[m].keys():
                if s not in self.data[m].keys():
                    self.data[m][s] = other_data[m][s]
                    continue
                for e in other_data[m][s].keys():
                    if e not in self.data[m][s].keys():
                        self.data[m][s][e] = other_data[m][s][e]
                        continue
                    self.data[m][s][e].extend(other_data[m][s][e])

    def compute_auc(self, upper, lower=0.0, metrics=None, splits=None, epochs=None, nbins=1000):
        assert lower < upper, "Lower limit must be strictly smaller than the upper limit!"
        query = self.get(metrics, splits, epochs, squeeze=False)

        for m, s_dict in query.items():
            for s, e_dict in s_dict.items():
                for e, val in e_dict.items():
                    #print("m: {}, s: {}, e: {}".format(m,s,e))
                    finite_val = val[np.isfinite(val)]
                    #print("finite_val:\n{}".format(finite_val))
                    hist, bin_edges = np.histogram(finite_val, bins=nbins, range=(lower, upper))
                    plot_y = np.cumsum(hist)
                    #print("plot_y:\n{}".format(plot_y))
                    plot_y = np.concatenate([[0], plot_y], axis=0)
                    plot_y = plot_y / val.size
                    #print("plot_y:\n{}".format(plot_y))
                    auc = np.trapz(plot_y, bin_edges)
                    #print("auc:\n{}".format(auc))
                    query[m][s][e] = auc / float(upper)
        return query


    def plot(self, outputpath, metrics=None, splits=None, epochs=None, gridFormat=16/9, plotly=False, matplotlib=False):
        if plotly:
            self._plot_training_plotly(outputpath, metrics=metrics, splits=splits, epochs=epochs, gridFormat=gridFormat)
        if matplotlib:
            self._plot_training_plt(outputpath, metrics=metrics, splits=splits, epochs=epochs)


    def plot_histogram(self, outputpath, metrics=None, splits=None, epochs=None, cumulative=True, gridFormat=16/9, nbins=512, plotly=False, matplotlib=False):
        if plotly:
            self._plot_histogram_plotly(outputpath, metrics=metrics, splits=splits, epochs=epochs, cumulative=cumulative, gridFormat=gridFormat, nbins=nbins)
        if matplotlib:
            self._plot_histogram_plt(outputpath, metrics=metrics, splits=splits, epochs=epochs, cumulative=cumulative, nbins=nbins)


    def _plot_training_plt(self, outputpath, metrics=None, splits=None, epochs=None):
        # outputpath: filepath where the plot should be saved
        # metrics can be a list of metrics names, or None if all metrics should be plotted
        color_dict = {}
        query = self.means_per_epoch(metrics, splits, epochs, squeeze=False)

        os.makedirs(outputpath, exist_ok=True)

        # Compute traces
        for idx, (m, s_dict) in enumerate(query.items()):
            f, ax = plt.subplots()
            ax.set_title(m)
            ax.set_xlabel("Epoch")
            axis_scale = 1.0
            if m in self.metrics.keys():
                meta = self.metrics[m]
            if meta is not None:
                if "axis_label" in meta.keys():
                    ax.set_ylabel(meta["axis_label"])
                if "axis_limits" in meta.keys():
                    ax.set_ylim(*meta["axis_limits"])
                if "axis_scale" in meta.keys():
                    axis_scale = float(meta["axis_scale"])

            for s, e_dict in s_dict.items():
                if s not in color_dict.keys():
                    trace_color = COLORS[len(color_dict.keys()) % len(COLORS)]
                    color_dict[s] = trace_color
                    showInLegend = True
                else:
                    trace_color = color_dict[s]
                    showInLegend = False

                y_values = list(e_dict.values())
                y_values = [y * axis_scale for y in y_values]

                ax.plot(
                    x=list(e_dict.keys()),
                    y=y_values,
                    c=trace_color,
                    label=s,
                )
            tikzplotlib.clean_figure()
            tikzplotlib.save(os.path.join(outputpath, "{}.tex".format(m)))
            plt.close()


    def _plot_training_plotly(self, outputpath, metrics=None, splits=None, epochs=None, gridFormat=16/9):
        # outputpath: filepath where the plot should be saved
        # metrics can be a list of metrics names, or None if all metrics should be plotted
        # if plotly is true, an interactive plot is generated using plotly. if plotly is false, matplotlib is used
        query = self.means_per_epoch(metrics, splits, epochs, squeeze=False)
        # Compute optimal cols and rows for given format
        metric_names = list(query.keys())
        cols = math.ceil(math.sqrt(len(metric_names) * gridFormat))
        rows = math.ceil(len(metric_names) / cols)
        fig = pyplots.make_subplots(
            rows=rows, cols=cols, subplot_titles=tuple(metric_names)
        )
        color_dict = {}

        # Compute traces
        metric_traces = {}
        for idx, (m, s_dict) in enumerate(query.items()):
            meta = None
            axis_scale = 1.0
            if m in self.metrics.keys():
                meta = self.metrics[m]
                if "axis_scale" in meta.keys():
                    axis_scale = float(meta["axis_scale"])

            metric_traces[m] = []
            for s, e_dict in s_dict.items():
                if s not in color_dict.keys():
                    trace_color = COLORS[len(color_dict.keys()) % len(COLORS)]
                    color_dict[s] = trace_color
                    showInLegend = True
                else:
                    trace_color = color_dict[s]
                    showInLegend = False

                y_values = list(e_dict.values())
                y_values = [y * axis_scale for y in y_values]

                trace = go.Scatter(
                    x=list(e_dict.keys()),
                    y=y_values,
                    mode="lines",
                    name=s,
                    legendgroup=s,
                    showlegend=showInLegend,
                    hoverinfo='all',
                    hovertemplate='(%{x}, %{y:0.4})',
                    line=dict(color="rgb({},{},{})".format(*trace_color)),
                )
                metric_traces[m].append(trace)

        # Generate plots
        row_idx = 1
        col_idx = 1
        for metric_idx, metric_name in enumerate(metric_names):
            meta = None
            if metric_name in self.metrics.keys():
                meta = self.metrics[metric_name]
            if meta is not None and "axis_label" in meta.keys():
                fig.update_yaxes(title_text=meta["axis_label"], row=row_idx, col=col_idx)
            if meta is not None and "axis_limits" in meta.keys():
                fig.update_yaxes(range=list(meta["axis_limits"]), row=row_idx, col=col_idx)
            # Add traces to subplot
            traces = metric_traces[metric_name]
            for trace in traces:
                fig.append_trace(trace, row=row_idx, col=col_idx)
            # Update row,col indices
            col_idx += 1
            if col_idx > cols:
                col_idx = 1
                row_idx += 1
        fig.update_layout(title_text="Trends w.r.t. Epoch")
        py.plot(fig, filename=outputpath, auto_open=False)


    def _plot_histogram_plt(self, outputpath, metrics=None, splits=None, epochs=None, cumulative=True, nbins=512):
        """TODO"""

        os.makedirs(outputpath, exist_ok=True)
        color_dict = {}
        query = self.get(metrics, splits, epochs, squeeze=False)

        # Compute traces
        for idx, (m, s_dict) in enumerate(query.items()):
            f, ax = plt.subplots()
            #ax.set_title("Cumulative Fraction of Samples w.r.t. Error Threshold")
            ax.grid(True)
            ax.tick_params(grid_alpha=0.25)
            ax.set_ylabel("Fraction of Dataset (%)")
            ax.set_ylim(bottom=0.0, top=100.0)
            ax.set_yticks(np.linspace(0.0, 100.0, 11))
            axis_scale = 1.0
            if m in self.metrics.keys():
                meta = self.metrics[m]
            if meta is not None:
                if "axis_label" in meta.keys():
                    ax.set_xlabel(meta["axis_label"])
                if "axis_limits" in meta.keys():
                    ax.set_xlim(*meta["axis_limits"])
                if "axis_scale" in meta.keys():
                    axis_scale = float(meta["axis_scale"])

            for s, e_dict in s_dict.items():
                for e, val in e_dict.items():
                    finite_val = val[np.isfinite(val)] * axis_scale
                    upper_limit = finite_val.max()
                    if finite_val.min() < 2000.0 and upper_limit > 2000.0:
                        upper_limit = 2000.0
                    hist, bin_edges = np.histogram(finite_val, bins=nbins, range=(finite_val.min(), upper_limit))
                    plot_y = np.cumsum(hist)
                    plot_y = np.concatenate([[0], plot_y], axis=0)
                    plot_y = 100.0 * plot_y / val.size # in percent

                    trace_name = "{}_epoch{:04d}".format(s, e)
                    if trace_name not in color_dict.keys():
                        trace_color = COLORS[len(color_dict.keys()) % len(COLORS)]
                        color_dict[trace_name] = trace_color
                        showInLegend = True
                    else:
                        trace_color = color_dict[trace_name]
                        showInLegend = False

                    name = trace_name
                    # if name.startswith("pvnet"):
                    #     name = "PVNet"
                    # elif name.startswith("combined"):
                    #     name = "Ours"
                    # elif name.startswith("handobject"):
                    #     name = "HandObjectNet"

                    ax.plot(
                        bin_edges,
                        plot_y,
                        c=trace_color,
                        label=name,
                    )
                    showInLegend = False
            ax.legend(loc='lower right')
            #tikzplotlib.clean_figure()
            tikzplotlib.save(os.path.join(outputpath, "{}.tex".format(m)))
            plt.savefig(os.path.join(outputpath, "{}.png".format(m)))
            plt.close()


    def _plot_histogram_plotly(self, outputpath, metrics=None, splits=None, epochs=None, cumulative=True, gridFormat=16/9, nbins=512):
        """TODO"""
        query = self.get(metrics, splits, epochs, squeeze=False)
        # Compute optimal cols and rows for given format
        metric_names = list(query.keys())
        cols = math.ceil(math.sqrt(len(metric_names) * gridFormat))
        rows = math.ceil(len(metric_names) / cols)
        fig = pyplots.make_subplots(
            rows=rows, cols=cols, subplot_titles=tuple(metric_names)
        )
        color_dict = {}

        # Compute traces
        metric_traces = {}
        for m, s_dict in query.items():
            meta = None
            axis_scale = 1.0
            if m in self.metrics.keys():
                meta = self.metrics[m]
                if "axis_scale" in meta.keys():
                    axis_scale = float(meta["axis_scale"])
            metric_traces[m] = []
            for s, e_dict in s_dict.items():
                for e, val in e_dict.items():
                    finite_idxs = np.isfinite(val)
                    if np.sum(finite_idxs) < 2:
                        continue
                    finite_val = val[np.isfinite(val)] * axis_scale
                    upper_limit = finite_val.max()
                    if finite_val.min() < 2000.0 and upper_limit > 2000.0:
                        upper_limit = 2000.0
                    hist, bin_edges = np.histogram(finite_val, bins=nbins, range=(finite_val.min(), upper_limit))
                    plot_y = np.cumsum(hist)
                    plot_y = np.concatenate([[0], plot_y], axis=0)
                    plot_y = plot_y / val.size

                    trace_name = "{}_epoch{:04d}".format(s, e)
                    if trace_name not in color_dict.keys():
                        trace_color = COLORS[len(color_dict.keys()) % len(COLORS)]
                        color_dict[trace_name] = trace_color
                        showInLegend = True
                    else:
                        trace_color = color_dict[trace_name]
                        showInLegend = False

                    trace = go.Scatter(
                        x=bin_edges, # ignore leftmost bin edge
                        y=plot_y,
                        mode='lines',
                        name=trace_name,
                        legendgroup=trace_name,
                        showlegend=showInLegend,
                        hoverinfo='all',
                        hovertemplate='(%{x:0.4f}, %{y:0.1%})',
                        line=dict(color="rgb({},{},{})".format(*trace_color))
                    )
                    metric_traces[m].append(trace)
                    showLegend = False

        # Generate plots
        row_idx = 1
        col_idx = 1
        for metric_idx, metric_name in enumerate(metric_names):
            meta = None
            if metric_name in self.metrics.keys():
                meta = self.metrics[metric_name]
            # Layout subplot
            if meta is not None:
                if "axis_label" in meta.keys():
                    fig.update_xaxes(title_text=meta["axis_label"], row=row_idx, col=col_idx)
                if "axis_limits" in meta.keys():
                    fig.update_xaxes(range=list(meta["axis_limits"]), row=row_idx, col=col_idx)
            fig.update_yaxes(range=[-0.05, 1.05], tickformat=",.0%", dtick=0.2, row=row_idx, col=col_idx)
            # Add traces to subplot
            traces = metric_traces[metric_name]
            for trace in traces:
                fig.append_trace(trace, row=row_idx, col=col_idx)
            # Update row,col indices
            col_idx += 1
            if col_idx > cols:
                col_idx = 1
                row_idx += 1
        fig.update_layout(title_text="Cumulative Fraction of Dataset w.r.t. Error Threshold")
        py.plot(fig, filename=outputpath, auto_open=False)


def _add_prefix_to_splits(d, prefix):
    result = {}
    for metric_name, split_dict in d.items():
        result[metric_name] = {}
        for split_name, epoch_dict in d[metric_name].items():
            result[metric_name]["{}{}".format(prefix, split_name)] = epoch_dict
    return result