
import re
from monitoring_sync.tracker import Tracker
from typing import List, Tuple, Any, Dict
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from flwr.common import log 
from logging import DEBUG
from flwr.server.history import History


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
        self.make_global_visualizations(config_specific_folder_name=folder_name)    

    def make_centralized_metrics_plot(self, folder_name: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        # Make four subplots in one representing centralized accuracy, loss, precision, recall and save it as a png
        timestamps_centralized = [ts for ts, _ in self.tracker.history.metrics_centralized_async['accuracy']]
        first_ts = timestamps_centralized[0]
        timestamps_centralized = [t - first_ts for t in timestamps_centralized]

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(timestamps_centralized, extract_vals_from_metrics(self.tracker.history.metrics_centralized_async['accuracy']))
        axs[0, 0].set_title('Centralized Accuracy')
        axs[0, 0].set_xlabel('Timestamp')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 1].plot(timestamps_centralized, extract_vals_from_metrics(self.tracker.history.losses_centralized_async))
        axs[0, 1].set_title('Centralized Loss')
        axs[0, 1].set_xlabel('Timestamp')
        axs[0, 1].set_ylabel('Loss')
        axs[1, 0].plot(timestamps_centralized, extract_vals_from_metrics(self.tracker.history.metrics_centralized_async['precision']))
        axs[1, 0].set_title('Centralized Precision')
        axs[1, 0].set_xlabel('Timestamp')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 1].plot(timestamps_centralized, extract_vals_from_metrics(self.tracker.history.metrics_centralized_async['recall']))
        axs[1, 1].set_title('Centralized Recall')
        axs[1, 1].set_xlabel('Timestamp')
        axs[1, 1].set_ylabel('Recall')
        # Set title for all subplots
        fig.suptitle(
            f'Centralized Metrics for setting:\n{self.tracker.dataset_name}, train_time: {self.tracker.total_train_time}, epochs:{self.tracker.epochs},\npartitioning:{self.tracker.partitioning}, alpha:{self.tracker.alpha}\n' + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, max_workers:{self.tracker.max_workers}, waiting_interval:{self.tracker.waiting_interval}')
        plt.savefig('results/' + folder_name + '/centralized_metrics.png')
        plt.clf()

    def make_centralized_final_perclass_metrics_plot(self, folder_name: str):
        # Make a two subplots for final precision and recall per class and save it as a png
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        x_axis = range(1, self.tracker.num_classes + 1)
        axs[0].bar(x_axis, self.tracker.history.metrics_centralized_async['precision_perclass'][-1][1])
        axs[0].set_title('Final Precision per class')
        axs[0].set_xlabel('Class')
        axs[0].set_ylabel('Precision')
        axs[1].bar(x_axis, self.tracker.history.metrics_centralized_async['recall_perclass'][-1][1])
        axs[1].set_title('Final Recall per class')
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Recall')
        fig.suptitle(f'Final Metrics for setting:\n{self.tracker.dataset_name}, train_time: {self.tracker.total_train_time}, epochs:{self.tracker.epochs},\npartitioning:{self.tracker.partitioning}, alpha:{self.tracker.alpha}' + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, max_workers:{self.tracker.max_workers}, waiting_interval:{self.tracker.waiting_interval}')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('results/' + folder_name
                    + '/centralized_final_perclass_metrics.png')
        plt.clf()

    def make_target_counts_plot(self, folder_name: str):
        # Create a stacked barplot that shows target_counts for each client (one bar = one client)
        fig, ax = plt.subplots(figsize=(15, 10))
        df = pd.DataFrame(self.tracker.target_counts)
        df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Final Metrics for setting:\n{self.tracker.dataset_name},  train_time: {self.tracker.total_train_time}, epochs:{self.tracker.epochs},\npartitioning:{self.tracker.partitioning}, alpha:{self.tracker.alpha}'  + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, max_workers:{self.tracker.max_workers}, waiting_interval:{self.tracker.waiting_interval}')
        ax.set_xlabel('Client')
        ax.set_ylabel('Count')
        lgd = ax.legend(title='Class', bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)

        plt.savefig('results/' + folder_name + '/target_counts.png',
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()


    def make_accuracies_per_client_over_time_plot(self, folder_name: str, title_suffix: str = ''):
        samples_per_client = self.tracker.sample_sizes
        n = self.tracker.num_clients
        # Count the number of times each client was trained
        times_started = np.zeros(n)
        for i in range(self.tracker.num_clients):
            if str(i) in self.tracker.history.metrics_distributed_fit_async['accuracy']:
                times_started[i] = len(self.tracker.history.metrics_distributed_fit_async['accuracy'][str(i)])

        num_rows = 3
        num_cols = np.ceil(n / num_rows).astype(int)
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(
            5 * num_cols, 5 * num_rows), constrained_layout=True)
        
        if num_cols == 1:
            axs.reshape(num_rows,1)
        fig.suptitle(f'Accuracies per client over time for setting:\n {self.tracker.dataset_name},  train_time: {self.tracker.total_train_time}, epochs:{self.tracker.epochs},\npartitioning:{self.tracker.partitioning}, alpha:{self.tracker.alpha}'  + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, max_workers:{self.tracker.max_workers}, waiting_interval:{self.tracker.waiting_interval}')
        # Flatten the axs array for easy indexing in case of a 2D configuration
        axs = axs.flatten()
        
        # Loop through the number of plots
        for i in range(n):
            if str(i) in self.tracker.history.metrics_distributed_fit_async['accuracy']:
                timestamps = [ts for ts,_ in self.tracker.history.metrics_distributed_fit_async['accuracy'][str(i)]]
                first_ts = timestamps[0]
                timestamps = [t - first_ts for t in timestamps]
                accuracies = [val for _,val in self.tracker.history.metrics_distributed_fit_async['accuracy'][str(i)]]
            else:
                timestamps = []
                accuracies = []
            axs[i].plot(timestamps, accuracies)
            axs[i].set_title(
                f'Client: {i+1}, Samples: {samples_per_client[i]}, Times started: {times_started[i]}')
            axs[i].set_ylim(0, 1)
            axs[i].set_ylabel('accuracy')
            axs[i].set_xlabel('time')
            # axs[i].set_ylim(0, max_sample_size)

        # Hide the remaining empty subplots
        for i in range(n, num_rows * num_cols):
            axs[i].axis('off')

        # Save the figure
        fig.savefig('results/' + folder_name + '/accuracies_per_client_over_time'+title_suffix+'.png')
        plt.clf()

    def make_precisions_per_client_over_time_plot(self, folder_name: str, title_suffix: str = ''):
        samples_per_client = self.tracker.sample_sizes
        n = self.tracker.num_clients
        # Count the number of times each client was trained
        times_started = np.zeros(n)
        for i in range(self.tracker.num_clients):
            if str(i) in self.tracker.history.metrics_distributed_fit_async['precision']:
                times_started[i] = len(self.tracker.history.metrics_distributed_fit_async['precision'][str(i)])

        num_rows = 3
        num_cols = np.ceil(n / num_rows).astype(int)
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(
            5 * num_cols, 5 * num_rows), constrained_layout=True)
        
        if num_cols == 1:
            axs.reshape(num_rows,1)
        fig.suptitle(f'Precisions per client over timefor setting:\n{self.tracker.dataset_name},  train_time: {self.tracker.total_train_time}, epochs:{self.tracker.epochs},\npartitioning:{self.tracker.partitioning}, alpha:{self.tracker.alpha}'  + \
            f'async_aggregation_strategy:{self.tracker.async_aggregation_strategy}, max_workers:{self.tracker.max_workers}, waiting_interval:{self.tracker.waiting_interval}')
        # Flatten the axs array for easy indexing in case of a 2D configuration
        axs = axs.flatten()

        # Loop through the number of plots
        for i in range(n):
            if str(i) in self.tracker.history.metrics_distributed_fit_async['precision']:
                timestamps = [ts for ts,_ in self.tracker.history.metrics_distributed_fit_async['precision'][str(i)]]
                first_ts = timestamps[0]
                timestamps = [t - first_ts for t in timestamps]
                precisions = [val for _,val in self.tracker.history.metrics_distributed_fit_async['precision'][str(i)]]
            else:
                timestamps = []
                precisions = []
            axs[i].plot(timestamps, precisions)
            axs[i].set_title(
                f'Client: {i+1}, Samples: {samples_per_client[i]}, Times started: {times_started[i]}')
            axs[i].set_ylim(0, 1)
            axs[i].set_ylabel('precision')
            axs[i].set_xlabel('time')
            # axs[i].set_ylim(0, max_sample_size)

        # Hide the remaining empty subplots
        for i in range(n, num_rows * num_cols):
            axs[i].axis('off')

        # Save the figure
        fig.savefig('results/' + folder_name + '/precisions_per_client_over_time'+title_suffix+'.png')
        plt.clf()

        

        
    def make_config_specific_visualizations(self, folder_name: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        # Make four subplots in one representing centralized accuracy, loss, precision, recall and save it as a png
        self.make_centralized_metrics_plot(folder_name)
        self.make_accuracies_per_client_over_time_plot(folder_name)
        self.make_precisions_per_client_over_time_plot(folder_name)
        self.make_centralized_final_perclass_metrics_plot(folder_name)
        self.make_target_counts_plot(folder_name)


    def make_global_visualizations(self, config_specific_folder_name: str = None):
        other_folders = [f for f in os.listdir(
            'results') if os.path.isdir(os.path.join('results', f))]
        other_trackers : List[Tracker] = []
        for other_folder in other_folders:
            with open('results/' + other_folder + '/result.pkl', 'rb') as f:
                other_tracker : Tracker = pickle.load(f)
                other_trackers.append(other_tracker)
        take_folder_name = 'take_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        configs = [{"dataset_name": ot.dataset_name,
                    "total_train_time": ot.total_train_time,
                    "epochs": ot.epochs,
                    "batch_size": ot.batch_size,
                    "learning_rate": ot.learning_rate,
                    "partitioning": ot.partitioning,
                    "num_clients": ot.num_clients,
                    "max_workers": ot.max_workers,
                } for ot in other_trackers]
        unique_configs = [i for n, i in enumerate(configs)
            if i not in configs[:n]]
        log(DEBUG, "unique_configs: %s", unique_configs)
        varied_params = {
            # 'partitioning': 'part' , 
            # 'alpha': 'a', 
            # 'num_clients': 'n', 
            'async_aggregation_strategy': 's' , 
            # 'max_workers': 'mw', 
            # 'waiting_interval': 'wi',
            }
        for unique_config in unique_configs:
            other_trackers_same_config = [ot for ot in other_trackers if ({"dataset_name": ot.dataset_name,
                    "total_train_time": ot.total_train_time,
                    "epochs": ot.epochs,
                    "batch_size": ot.batch_size,
                    "learning_rate": ot.learning_rate,
                    "partitioning": ot.partitioning,
                    "num_clients": ot.num_clients,
                    "max_workers": ot.max_workers,
                } == unique_config) and ot.is_async]
            log(DEBUG,
                f"Number of trackers with the same config: {len(other_trackers_same_config)}")
            self.make_comparative_plots(
                other_trackers_same_config, unique_config=unique_config, take_folder_name=take_folder_name, config_specific_folder_name=config_specific_folder_name, varied_parameters=varied_params)

    def make_comparative_plots(self, all_trackers: List["Tracker"], take_folder_name: str, folder_name: str = None, unique_config: Dict = None, 
                               config_specific_folder_name: str = None, varied_parameters: Dict[str,str] = None):
        varied_parameters_local = varied_parameters.copy()

        if 'partitioning' in varied_parameters_local.keys() and varied_parameters_local['partitioning'] == 'iid' and 'alpha' in varied_parameters_local.keys():
            varied_parameters_local.pop('alpha')
        if not folder_name:
            folder_name = 'async_config_' + config_to_str(unique_config) 
        if len(all_trackers) == 0:
            return

        # make sure that folder 'global' exists
        if not os.path.exists('global'):
            os.makedirs('global')
        if not os.path.exists('global/' + take_folder_name):
            os.makedirs('global/' + take_folder_name)
        if not os.path.exists('global/' + take_folder_name + '/' + folder_name):
            os.makedirs('global/' + take_folder_name + '/' + folder_name)

        # Make comparison plots for accuracy, loss, precision, recall
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        for tracker in all_trackers:
            linewidth = 1.0 if tracker.partitioning != 'iid' else 3.0
            label = get_label_from_varied_params(varied_params=varied_parameters_local,tracker=tracker) # (tracker.partitioning if 'dirichlet' not in tracker.partitioning else tracker.partitioning + \
                # '(' + str(tracker.alpha) + ')') + ', n=' + str(tracker.num_clients) + ', \n s=' + tracker.async_aggregation_strategy + ', mw=' + str(tracker.max_workers) + ', wi=' + str(tracker.waiting_interval)
            timestamps = [ts for ts, _ in tracker.history.metrics_centralized_async['accuracy']]
            first_ts = timestamps[0]
            timestamps = [t - first_ts for t in timestamps]
            axs[0, 0].plot(timestamps, extract_vals_from_metrics(tracker.history.metrics_centralized_async['accuracy']),
                           label=label, linewidth=linewidth)
            axs[0, 1].plot(timestamps, extract_vals_from_metrics(tracker.history.losses_centralized_async), 
                           label=label, linewidth=linewidth)
            axs[1, 0].plot(timestamps, extract_vals_from_metrics(tracker.history.metrics_centralized_async['precision']),
                           label=label, linewidth=linewidth)
            axs[1, 1].plot(timestamps, extract_vals_from_metrics(tracker.history.metrics_centralized_async['recall']),
                           label=label, linewidth=linewidth)
        axs[0, 0].set_title('Centralized Accuracy')
        axs[0, 0].set_xlabel('time')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 1].set_title('Centralized Loss')
        axs[0, 1].set_xlabel('time')
        axs[0, 1].set_ylabel('Loss')
        axs[1, 0].set_title('Centralized Precision')
        axs[1, 0].set_xlabel('time')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 1].set_title('Centralized Recall')
        axs[1, 1].set_xlabel('time')
        axs[1, 1].set_ylabel('Recall')
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        fig.suptitle(
            f'Centralized metrics comparison:\n' + config_to_str(unique_config))
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('global/' + take_folder_name + '/'
                    + folder_name + '/centralized_metrics_comparison.png')
        plt.clf()

        # Also compare only to iid
        if self.tracker.partitioning != 'iid':
            iid_tracker = [
                tracker for tracker in all_trackers if tracker.partitioning == 'iid']
            if len(iid_tracker) > 0:
                iid_tracker = iid_tracker[0]
                fig, axs = plt.subplots(2, 2, figsize=(20, 10))
                for tracker in [iid_tracker, self.tracker]:
                    label =  get_label_from_varied_params(varied_params=varied_parameters_local,tracker=tracker)
                    timestamps = [ts for ts, _ in tracker.history.metrics_centralized_async['accuracy']]
                    first_ts = timestamps[0]
                    timestamps = [t - first_ts for t in timestamps]
                    axs[0, 0].plot(timestamps, extract_vals_from_metrics(tracker.history.metrics_centralized_async['accuracy']),
                           label=label, linewidth=linewidth)
                    axs[0, 1].plot(timestamps, extract_vals_from_metrics(tracker.history.losses_centralized_async), 
                                label=label, linewidth=linewidth)
                    axs[1, 0].plot(timestamps, extract_vals_from_metrics(tracker.history.metrics_centralized_async['precision']),
                                label=label, linewidth=linewidth)
                    axs[1, 1].plot(timestamps, extract_vals_from_metrics(tracker.history.metrics_centralized_async['recall']),
                                label=label, linewidth=linewidth)
                axs[0, 0].set_title('Centralized Accuracy')
                axs[0, 0].set_xlabel('time')
                axs[0, 0].set_ylabel('Accuracy')
                axs[0, 1].set_title('Centralized Loss')
                axs[0, 1].set_xlabel('time')
                axs[0, 1].set_ylabel('Loss')
                axs[1, 0].set_title('Centralized Precision')
                axs[1, 0].set_xlabel('time')
                axs[1, 0].set_ylabel('Precision')
                axs[1, 1].set_title('Centralized Recall')
                axs[1, 1].set_xlabel('time')
                axs[1, 1].set_ylabel('Recall')
                axs[0, 0].legend()
                axs[0, 1].legend()
                axs[1, 0].legend()
                axs[1, 1].legend()
                fig.suptitle(
                    f'Centralized metrics comparison:\n' + config_to_str(unique_config))
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.savefig('results/' + config_specific_folder_name
                            + '/centralized_metrics_comparison_to_iid_only.png')
                plt.clf()

        # In len(other_trackers) + 1 subplots show the sample sizes of each tracker (one subplot = one tracker) x-axis->clients, y-axis->sample sizes
        # Order the subplots in two rows
        plt.tight_layout()
        cols = len(all_trackers) // 2
        if len(all_trackers) % 2 == 1:
            cols += 1
        fig, axs = plt.subplots(2, cols , figsize=(20, 10))
        if cols == 1:
            axs = axs.reshape(2, 1)
        clients = range(1, all_trackers[0].num_clients + 1)
        for i, tracker in enumerate(all_trackers):
            label =  get_label_from_varied_params(varied_params=varied_parameters_local,tracker=tracker)
            axs[i % 2, i // 2].bar(clients, tracker.sample_sizes, label=label)
            axs[i % 2, i // 2].set_title('Sample sizes per client\n' + label)
            axs[i % 2, i // 2].set_xlabel('Client')
            axs[i % 2, i // 2].set_ylabel('Sample size')

        # Delete the empty subplot if there is one
        for i in range(len(all_trackers), 2 * cols):
            axs[i % 2, i // 2].axis('off')

        fig.suptitle(
            f'Sample Sizes comparison\n{config_to_str(unique_config)}')
        plt.tight_layout(pad=1, w_pad=2, h_pad=3.0)
        plt.savefig('global/' + take_folder_name + '/'
                    + folder_name + '/sample_sizes_comparison.png')
        plt.clf()

        # Do the same for target_counts
        cols = len(all_trackers) // 2
        if len(all_trackers) % 2 == 1:
            cols += 1
        fig, axs = plt.subplots(2, cols, figsize=(20, 10))
        clients = range(1, all_trackers[0].num_clients + 1)
        if cols == 1:
            axs = axs.reshape(2, 1)
        
        for i, tracker in enumerate(all_trackers):
            label =  get_label_from_varied_params(varied_params=varied_parameters_local,tracker=tracker)

            df = pd.DataFrame(tracker.target_counts)
            df.plot(kind='bar', stacked=True, ax=axs[i % 2, i // 2], legend=False)
            axs[i % 2, i
                // 2].set_title('Class and sample distribution per client\n' + label)
            axs[i % 2, i // 2].set_xlabel('Client')
            axs[i % 2, i // 2].set_ylabel('Count')

        # Add one legend for all subplots
        handles, labels = axs[len(all_trackers) % 2, (len(
            all_trackers) // 2) - 1].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, title='Class',
                         bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        for i in range(len(all_trackers), 2 * cols):
            axs[i % 2, i // 2].axis('off')

        fig.suptitle(
            f'Target counts comparison:\n{config_to_str(unique_config)}')
        plt.tight_layout(pad=1, w_pad=2, h_pad=3.0)
        fig.savefig('global/' + take_folder_name + '/' + folder_name
                    + '/target_counts_comparison.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()

        fig, axs = plt.subplots(2, cols, figsize=(20, 10))
        classes = range(all_trackers[0].num_classes)
        if cols == 1:
            axs = axs.reshape(2, 1)
        for i, tracker in enumerate(all_trackers):
            label =  get_label_from_varied_params(varied_params=varied_parameters_local,tracker=tracker)
            samples_per_class = np.zeros(tracker.num_classes)
            for cid in tracker.history.metrics_distributed_fit_async['accuracy'].keys():
                times_chosen = len(tracker.history.metrics_distributed_fit_async['accuracy'][cid])
                samples_per_class = samples_per_class + tracker.target_counts[int(cid)] * times_chosen

            axs[i % 2, i // 2].bar(classes, samples_per_class, label=label)
            axs[i % 2, i // 2].set_title('Total number of samples used per class\n' + label)
            axs[i % 2, i // 2].set_xlabel('Class')
            axs[i % 2, i // 2].set_ylabel('Count')
        for i in range(len(all_trackers), 2 * cols):
            axs[i % 2, i // 2].axis('off')

        fig.suptitle(
            f'Total number of samples per class comparison:\n{config_to_str(unique_config)}')
        plt.tight_layout(pad=1, w_pad=2, h_pad=3.0)
        fig.savefig('global/' + take_folder_name + '/' + folder_name
                    + '/total_samples_per_class_comparison.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()

        # Make a comparison plot for final global train times in one bar plot without subplots
        fig, ax = plt.subplots(figsize=(15, 10))
        global_times = [
            other_tracker.global_total_train_time for other_tracker in all_trackers]
        labels = [get_label_from_varied_params(varied_parameters_local, other_tracker) for other_tracker in all_trackers]
        ax.bar(labels, global_times)
        ax.set_title(
            f'Global train times comparison\n{config_to_str(unique_config)}')
        ax.set_xlabel('Partitioning')
        plt.xticks(rotation=45)
        ax.set_ylabel('Time (s)')
        plt.savefig('global/' + take_folder_name + '/'
                    + folder_name + '/global_train_times_comparison.png')
        plt.clf()

        # Make a comparison plot for final accuracies, losses, precisions and recalls as four subplots (barplots)
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        
        final_accuracies = [ extract_vals_from_metrics(other_tracker.history.metrics_centralized_async['accuracy'])[-1] for other_tracker in all_trackers]
        final_losses = [ extract_vals_from_metrics(other_tracker.history.losses_centralized_async)[-1] for other_tracker in all_trackers]
        final_precisions = [ extract_vals_from_metrics(other_tracker.history.metrics_centralized_async['precision'])[-1] for other_tracker in all_trackers]
        final_recalls = [ extract_vals_from_metrics(other_tracker.history.metrics_centralized_async['recall'])[-1] for other_tracker in all_trackers]
        labels = [get_label_from_varied_params(varied_parameters_local, other_tracker) for other_tracker in all_trackers]

        axs[0, 0].bar(labels, final_accuracies)
        axs[0, 0].set_title('Final Accuracies')
        axs[0, 0].set_xlabel('Partitioning')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_xticklabels(labels, rotation=45)
        axs[0, 1].bar(labels, final_losses)
        axs[0, 1].set_title('Final Losses')
        axs[0, 1].set_xlabel('Partitioning')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xticklabels(labels, rotation=45)
        axs[1, 0].bar(labels, final_precisions)
        axs[1, 0].set_title('Final Precisions')
        axs[1, 0].set_xlabel('Partitioning')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 0].set_xticklabels(labels, rotation=45)
        axs[1, 1].bar(labels, final_recalls)
        axs[1, 1].set_title('Final Recalls')
        axs[1, 1].set_xlabel('Partitioning')
        axs[1, 1].set_ylabel('Recall')
        axs[1, 1].set_xticklabels(labels, rotation=45)

        fig.suptitle(
            f'Comparison of final metrics:\n{config_to_str(unique_config)}')

        plt.tight_layout(pad=1, w_pad=2, h_pad=3.0)
        plt.savefig('global/' + take_folder_name + '/'
                    + folder_name + '/final_metrics_comparison.png')
        plt.clf()

        n_samples = [ 
                sum([
                    sum([samples for timestamp, samples in cid_dict
                         ]) for cid_dict in tracker.history.metrics_distributed_fit_async['sample_sizes'].values()
                ])
             for tracker in all_trackers ] 
        
        log(DEBUG, f"n_samples: {n_samples}")
        labels = [get_label_from_varied_params(varied_parameters_local, other_tracker) for other_tracker in all_trackers]
        plt.bar(labels, n_samples)
        plt.title(
            f'Total samples seen\n{config_to_str(unique_config)}')
        plt.xlabel('Partitioning')
        plt.xticks(rotation=45)
        plt.ylabel('Number of samples seen')
        plt.savefig('global/' + take_folder_name + '/'
                    + folder_name + '/total_samples_seen_comparison.png')
        plt.clf()
        

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