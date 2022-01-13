import numpy as np
import json
from datetime import datetime


def get_reward_array(hist_dict):
    """
    Get the reward array from the history dict.
    """
    hist = [val.get('reward') for val in hist_dict.values()]
    return np.array(hist)


def get_steps_array(hist_dict):
    """
    Get the steps array from the history dict.
    """
    hist = [val.get('steps') for val in hist_dict.values()]
    return np.array(hist)


def get_penalties_array(hist_dict):
    """
    Get the penalties array from the history dict.
    """
    hist = [val.get('penalties') for val in hist_dict.values()]
    return np.array(hist)


def get_average_reward_last_n(hist_dict, n_episodes):
    """
    Get the average reward of the last n_episodes episodes.
    """
    hist = [val.get('reward') for val in hist_dict.values()]
    return np.mean(hist[-n_episodes:])


def get_average_steps_last_n(hist_dict, n_episodes):
    """
    Get the average steps of the last n_episodes episodes.
    """
    hist = [val.get('steps') for val in hist_dict.values()]
    return np.mean(hist[-n_episodes:])


def get_average_penalties_last_n(hist_dict, n_episodes):
    """
    Get the average penalties of the last n_episodes episodes.
    """
    hist = [val.get('penalties') for val in hist_dict.values()]
    return np.mean(hist[-n_episodes:])


def generate_report_file(config, report, hist, qtable):
    """
    Generate a report from the history dict.
    """
    report_content = {}
    report_content['config'] = config
    report_content['report'] = report
    report_content['qtable'] = qtable.tolist()
    report_content['hist'] = hist

    report_file_name = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S") + '_report.json'
    json.dump(report_content, open(
        "resources/reports/"+report_file_name, "w"), indent=4)
    return
