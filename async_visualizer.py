import math
from monitoring_sync.tracker import Tracker
from typing import List, Tuple, Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flwr.common import log 
from logging import DEBUG
from flwr.server.history import History
import seaborn as sns
import glob

nice_goal_label_names = ['WALKING', 'SITTING', 'STANDING', 'LYING_DOWN', 'RUNNING', 'BICYCLING']


def extract_vals_from_metrics(metrics: List[Tuple[int, Any]]):
    return [val for _, val in metrics]

class AsyncVisualizer:
    """Visualize the collected metrics."""

    def __init__(self, tracker:Tracker) -> None:
        self.tracker = tracker


    def extract_vals_from_metrics(self, metrics: List[Tuple[int, Any]]):
        return [val for _, val in metrics]

    def plot(self, folder_name: str):
        self.make_config_specific_visualizations(folder_name)
        # self.make_global_visualizations(config_specific_folder_name=folder_name)    

    def extract_timestamps(self, metric_key):
        timestamps = [ts for ts, _ in self.tracker.history.metrics_centralized[metric_key]]
        first_ts = timestamps[0]
        return [t - first_ts for t in timestamps]

    def plot_metric(self, ax, x_values, y_values, title, xlabel, ylabel):
        """Plot a single metric on the provided Axes object."""
        ax.plot(x_values, y_values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_final_centralized_confusion_matrix(self, folder_name: str):
        # Make a heatmap for the final confusion matrix and save it as a png
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.tracker.history.metrics_centralized['confusion_matrix'][-1][1], annot=True, fmt='d', ax=ax, xticklabels=nice_goal_label_names, yticklabels=nice_goal_label_names)
        ax.set_title('Final Confusion Matrix')
        plt.savefig('results/' + folder_name + '/final_centralized_confusion_matrix.png')
        plt.clf()

    def make_centralized_metrics_plot(self, folder_name: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        timestamps = self.extract_timestamps('accuracy')  # Assuming all metrics share the same timestamps

        metrics = [
            # ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            # ('precision', 'Precision'),
            # ('recall', 'Recall'),
            ('f1', 'F1'),
            ('balanced_accuracy', 'Balanced Accuracy'),
        ]

        num_metrics = len(metrics)
        grid_size = math.ceil(math.sqrt(num_metrics))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
        axs = axs.flatten()

        for ax, (metric_key, metric_name) in zip(axs, metrics):
            metric_values = extract_vals_from_metrics(self.tracker.history.metrics_centralized.get(metric_key, []))
            self.plot_metric(ax, timestamps, metric_values, f'Centralized {metric_name}', 'Timestamp', metric_name)

        fig.suptitle(f'Centralized Metrics, asynchronous setting:\n{self.tracker.dataset_name}, train_time: {self.tracker.total_train_time},'+ \
                     f',\naugmentation:{self.tracker.data_augmentation} ' + \
                     f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, mixing alpha:{self.tracker.fedasync_mixing_alpha}',
                       fontsize=12)
        plt.savefig(f'results/{folder_name}/centralized_metrics.png')
        plt.clf()

    def make_centralized_final_perclass_metrics_plot(self, folder_name: str):
        # Make a two subplots for final precision and recall per class and save it as a png
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        x_axis = range(1, self.tracker.num_classes + 1)
        axs[0].bar(x_axis, self.tracker.history.metrics_centralized['precision_perclass'][-1][1])
        axs[0].set_title('Final Precision per class')
        axs[0].set_xlabel('Class')
        axs[0].set_ylabel('Precision')
        axs[1].bar(x_axis, self.tracker.history.metrics_centralized['recall_perclass'][-1][1])
        axs[1].set_title('Final Recall per class')
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Recall')
        axs[2].bar(x_axis, self.tracker.history.metrics_centralized['f1_perclass'][-1][1])
        axs[2].set_title('Final F1 per class')
        axs[2].set_xlabel('Class')
        axs[2].set_ylabel('F1')
        fig.suptitle(f'Final Metrics for setting:\n{self.tracker.dataset_name}, train_time: {self.tracker.total_train_time},\n' + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, Mixing_alpha: {self.tracker.fedasync_mixing_alpha}')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('results/' + folder_name
                    + '/centralized_final_perclass_metrics.png')
        plt.clf()

    def make_target_counts_plot(self, folder_name: str):
        # Create a stacked barplot that shows target_counts for each client (one bar = one client)
        fig, ax = plt.subplots(figsize=(15, 10))
        df = pd.DataFrame(self.tracker.target_counts)
        df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Target counts for setting:\n{self.tracker.dataset_name},  train_time: {self.tracker.total_train_time},\ndata augmentation:{self.tracker.data_augmentation}'  + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, mixing alpha:{self.tracker.fedasync_mixing_alpha}')
        ax.set_xlabel('Client')
        ax.set_ylabel('Count')
        lgd = ax.legend(title='Class', bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)

        plt.savefig('results/' + folder_name + '/target_counts.png',
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()


    def get_client_metrics(self, metric_name, client_id):
        """Extracts and normalizes timestamps and values for a specific client metric."""
        if str(client_id) in self.tracker.history.metrics_distributed_fit_async[metric_name]:
            timestamps = [ts for ts, _ in self.tracker.history.metrics_distributed_fit_async[metric_name][str(client_id)]]
            first_ts = timestamps[0]
            normalized_timestamps = [t - first_ts for t in timestamps]
            values = [val for _, val in self.tracker.history.metrics_distributed_fit_async[metric_name][str(client_id)]]
            return normalized_timestamps, values
        return [], []

    def count_times_started(self, metric_name):
        """Counts the number of times each client has started based on the metric."""
        times_started = np.zeros(self.tracker.num_clients)
        for i in range(self.tracker.num_clients):
            if str(i) in self.tracker.history.metrics_distributed_fit_async[metric_name]:
                times_started[i] = len(self.tracker.history.metrics_distributed_fit_async[metric_name][str(i)])
        return times_started

    def make_metrics_per_client_over_time_plot(self, folder_name: str, metric_name: str, title_suffix: str = ''):
        samples_per_client = self.tracker.sample_sizes
        times_started = self.count_times_started(metric_name)
        n = self.tracker.num_clients
        # Calculate number of rows and columns for a square or nearly square configuration
        num_cols = int(math.ceil(math.sqrt(n)))
        num_rows = int(math.ceil(n / num_cols))
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), constrained_layout=True)
        fig.suptitle(f'{metric_name.capitalize()} per client over time (before training) for setting:\n{self.tracker.dataset_name}, train_time: {self.tracker.total_train_time},\nasync_aggregation_strategy:{self.tracker.async_aggregation_strategy}' + title_suffix)
        axs = axs.flatten()

        for i in range(n):
            timestamps, values = self.get_client_metrics(metric_name, i)
            axs[i].plot(timestamps, values)
            axs[i].set_title(f'Client: {i+1}, Samples: {samples_per_client[i]}, Times started: {times_started[i]}')
            axs[i].set_ylim(0, 1)
            axs[i].set_ylabel(metric_name)
            axs[i].set_xlabel('time')

        for i in range(n, num_rows * num_cols):
            axs[i].axis('off')

        fig.savefig(f'results/{folder_name}/{metric_name.lower()}_per_client_over_time{title_suffix}.png')
        plt.clf()

    
    def make_interval_plot_async(self, folder_name: str):
        times_started = self.tracker.history.metrics_distributed_fit_async['time_started']
        local_train_times = self.tracker.history.metrics_distributed_fit_async['local_train_time']

        start_timestamp = self.tracker.history.metrics_centralized['start_timestamp'][0][1]
        end_timestamp = self.tracker.history.metrics_centralized['end_timestamp'][0][1]

        self.tracker.history.metrics_distributed_fit_async['time_started']

        df = pd.DataFrame([], columns=['cid', 'start', 'end'])

        appended_data = []
        for cid, lst in times_started.items():
            for i, (_, start_time) in enumerate(lst):
                local_train_time = local_train_times[cid][i][1]
                new_df = pd.DataFrame([[cid, start_time-start_timestamp, start_time + local_train_time -start_timestamp]] , columns=['cid', 'start', 'end'])
                appended_data.append(new_df)

        final_df = pd.concat(appended_data)
        df = final_df
        fig, ax = plt.subplots(figsize=(10, 8))
        clients = df['cid'].unique()
        client_positions = {cid: i for i, cid in enumerate(sorted(clients), start=1)}
        for _, row in df.iterrows():
            cid = row['cid']
            start = row['start']
            end = row['end']
            ax.plot([start, end], [client_positions[cid], client_positions[cid]], marker='o')

        ax.axvline(0, color='b', linestyle='--', label='Start Timestamp')
        ax.axvline(end_timestamp-start_timestamp, color='r', linestyle='--', label='End Timestamp')

        ax.legend()
        plt.yticks(list(client_positions.values()), list(client_positions.keys()))
        plt.xlabel('Time')
        plt.ylabel('Client ID')
        plt.title('Intervals Marked by Start and End Times for Each Client')
        plt.grid(True)
        plt.savefig('results/' + folder_name + '/intervals.png')
        plt.clf()

    def make_f1_over_time_heatmap(self, folder_name: str):
        if getattr(self.tracker, 'id', None) is not None:
            id = self.tracker.id
            # Load all .npy files from a directory
            npy_files = glob.glob(f'f1_scores/{id}/*.npy')
            npy_files.sort()
            f1_scores = np.array([np.array(np.load(file, allow_pickle=True), dtype=np.float32) for file in npy_files])
            plt.figure(figsize=(20,10))
            sns.heatmap(f1_scores)
            plt.title('F1_scores over time per client')
            plt.ylabel('Time')
            plt.xlabel('Client')
            plt.savefig(f'results/{folder_name}/f1_over_time_heatmap.png')
            plt.clf()
        
    def make_f1_for_running_class_and_other_classes(self, folder_name: str):
        f1_pc = np.array([np.array(f1s) for ts, f1s in self.tracker.history.metrics_centralized['f1_perclass']])
        sns.heatmap(f1_pc)
        plt.xticks(np.arange(6)+0.5, nice_goal_label_names, rotation=90)
        plt.ylabel('Time')
        plt.xlabel('Class')
        plt.title('Centralized per-class F1-Scores over time')
        plt.savefig(f'results/{folder_name}/f1_perclass_over_time.png')
        plt.clf()

        plt.plot(f1_pc[:, 4])
        plt.title('F1 score of the running class over time')
        plt.xlabel('Time')
        plt.ylabel('F1-Score')
        plt.savefig(f'results/{folder_name}/f1_running_class_over_time.png')
        plt.clf()

    def make_config_specific_visualizations(self, folder_name: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        # Make four subplots in one representing centralized accuracy, loss, precision, recall and save it as a png
        self.make_centralized_metrics_plot(folder_name)
        self.make_metrics_per_client_over_time_plot(folder_name, 'accuracy')
        self.make_metrics_per_client_over_time_plot(folder_name, 'precision')
        self.make_metrics_per_client_over_time_plot(folder_name, 'f1')
        self.make_centralized_final_perclass_metrics_plot(folder_name)
        self.make_target_counts_plot(folder_name)
        self.make_interval_plot_async(folder_name)
        self.plot_final_centralized_confusion_matrix(folder_name)
        self.make_f1_over_time_heatmap(folder_name)
        self.make_f1_for_running_class_and_other_classes(folder_name)


    
        

def get_label_from_varied_params(varied_params: Dict[str,str], tracker: Tracker):
    if len(varied_params) > 4:
        half = len(varied_params.keys()) // 2
        res = ', '.join([f'{varied_params[param]}={getattr(tracker, param)}' for param in list(varied_params.keys())[:half]])
        res += '\n'
        res += ', '.join([f'{varied_params[param]}={getattr(tracker, param)}' for param in list(varied_params.keys())[half:]])
    else:
        res = ', '.join([f'{varied_params[param]}={getattr(tracker, param)}' for param in varied_params.keys()])
    #if 'iid' in res:
    #    res = re.sub(r"a=[+-]?([0-9]*[.])?[0-9]+", "", res)
    return res

def config_to_str(config: Dict):
    return ', '.join([f'{key}={value}' for key, value in config.items()])
        # NOTE: If there are multiple values for the **same label** the plot will be drawn over the previous plot.