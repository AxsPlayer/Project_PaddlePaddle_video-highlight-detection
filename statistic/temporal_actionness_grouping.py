# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
The Temporal Actioness Grouping after outputting Neural Network result.

Authors: AxsPlayer
Date: 2018/04/17 12:49:06
"""
import numpy as np


# transform scores to interval
def check_start_or_end(index, score_list, score_cutoff):
    """Check the attribute of elements in list.

    Check the attribute of element whether it's the beginning
    or the ending of frame.

    Args:
        index: The index of the element in list.
        score_list: The score list which contains continuous actioness
            scores of each frame.
        score_cutoff: The cutoff set by user which filters the active
            frames.

    Returns:
        start_flag: 1 for beginning point.
        end_flag: 1 for ending point.
    """
    # Initial the start flag and end flag.
    start_flag, end_flag = 0, 0

    # Judge whether the element is beginning or the ending.
    if score_list[index] >= score_cutoff:
        if index == 0:  # Start of list.
            start_flag = 1
        if index == len(score_list) - 1:  # End of list.
            end_flag = 1
        if index - 1 >= 0:
            if score_list[index - 1] >= score_cutoff:
                start_flag = 0
            else:
                start_flag = 1
        if index + 1 <= len(score_list) - 1:
            if score_list[index + 1] >= score_cutoff:
                end_flag = 0
            else:
                end_flag = 1

    return start_flag, end_flag


def filter_frames(actioness_score, actioness_cutoff):
    """Filter the satisfying frames.

    Filter the satisfying frames using actionness cutoff.

    Args:
        actioness_score： The score list which contains continuous actioness
            scores of each frame.
        actioness_cutoff: The cutoff set by user which filters the active
            frames.

    Returns:
        frame_dictionary: A dictionary containing each frame's starting frame
            and ending frame.
    """
    # Initial the environment.
    frame_dictionary = {}
    frame_num = 0

    # Record starting and ending points into frame dictionary for each frame.
    for i in xrange(len(actioness_score)):
        # Fetch flags for index.
        start_flag, end_flag = check_start_or_end(i, actioness_score,
                                                  actioness_cutoff)
        # Set and update each frame's starting and ending points.
        if start_flag == 1:
            frame_dictionary[frame_num] = {'start': i}
        if end_flag == 1:
            frame_dictionary[frame_num]['end'] = i
            frame_num = frame_num + 1

    return frame_dictionary


def ave_score(start_num, end_num, actioness_score):
    """Calculate the average score.

    :param start_num: The start point.
    :param end_num: The end point.
    :param actioness_score: A series of actionness scores.

    :return: The average score of target frame.
    """
    ave_sum = round(sum(actioness_score[start_num:(end_num + 1)]) \
                    / (end_num - start_num + 1), 3)
    return ave_sum


def group_frames(frame_dictionary, maximum_portion, actioness_score):
    """Group frames into proposals.

    Group frames into proposals using maximum_portion.

    Args:
        frame_dictionary： A dictionary containing each frame's starting frame
            and ending frame.
        maximum_portion: The value set by user which is used to filter the valuable
            proposals whose inactive portion is under maximum portion.
        actioness_score： The score list which contains continuous actioness
            scores of each frame.

    Returns:
        proposal_dictionary: A dictionary which contains each proposal's starting
            frame and ending frame.
    """
    # Initial the environment.
    key_num = sorted(frame_dictionary.keys(), reverse=False)
    proposal_num = 0
    proposal_dictionary = {}
    i = 0

    # Circulation to find out the suitable proposals.
    while i < len(key_num) - 1:
        # Deal with cold starting of circulation.
        if proposal_num not in proposal_dictionary.keys():
            proposal_dictionary[proposal_num] = \
                {'start': frame_dictionary[key_num[i]]['start'],
                 'end': frame_dictionary[key_num[i]]['end'],
                 'blank': 0,
                 'score': ave_score(frame_dictionary[key_num[i]]['start'],
                                    frame_dictionary[key_num[i]]['end'], actioness_score)}
        else:
            # Calculate the length of blank period between frames and the length of total period of candidate.
            blank_period = float(frame_dictionary[key_num[i + 1]]['start']) - \
                           float(proposal_dictionary[proposal_num]['end']) + \
                           float(proposal_dictionary[proposal_num]['blank']) - 1
            total_period = float(frame_dictionary[key_num[i + 1]]['end']) - \
                           float(proposal_dictionary[proposal_num]['start']) + 1

            # Deteminate whether to accept the proposal or not according to maximum portion.
            if float(blank_period) / total_period <= maximum_portion:
                proposal_dictionary[proposal_num]['end'] = \
                    frame_dictionary[key_num[i + 1]]['end']
                proposal_dictionary[proposal_num]['blank'] = blank_period
                proposal_dictionary[proposal_num]['score'] = ave_score(proposal_dictionary[proposal_num]['start'],
                                                                       proposal_dictionary[proposal_num]['end'],
                                                                       actioness_score)
            else:
                proposal_num = proposal_num + 1
                proposal_dictionary[proposal_num] = \
                    {'start': frame_dictionary[key_num[i + 1]]['start'],
                     'end': frame_dictionary[key_num[i + 1]]['end'],
                     'blank': 0,
                     'score': ave_score(frame_dictionary[key_num[i]]['start'],
                                        frame_dictionary[key_num[i]]['end'], actioness_score)}
            i = i + 1  # Count number.

    return proposal_dictionary


def create_proposal(actioness_score, actioness_cutoff, maximum_portion):
    """Group frames into temporal proposal using actioness score.

    The method is firstly using actioness cutoff to filter some frames
    and using maximum portion to absorb other discrete frames, finally,
    the proposals are selected.

    Args:
        actioness_score: Continuous line of actioness score which represent
            each frame's brilliance.
        actioness_cutoff: The cutoff which is used to select the continuous
            frames.
        maximum_portion: The maximum portion of frames whose scores are below
            the actioness_cutoff in each proposal.

    Returns:
         proposal_dictionary: A dictionary which contains a series of proposals
            whose attributes are starting point and ending point.
    """
    # Filter frames and store frames into dictionary.
    frame_dictionary = filter_frames(actioness_score, actioness_cutoff)

    # Group frames into proposals.
    proposal_dictionary = group_frames(frame_dictionary, maximum_portion, actioness_score)

    # Output the results.
    return proposal_dictionary


def create_multiple_proposals(actioness_cutoff_ceiling, actioness_cutoff_floor, maximum_portion_ceiling,
                              maximum_portion_floor, actioness_cutoff_step, maximum_portion_step, repeat_ratio):
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    for actioness_cutoff in np.arange(actioness_cutoff_floor, actioness_cutoff_ceiling + 0.01, actioness_cutoff_step):
        for maximum_portion in np.arange(maximum_portion_floor, maximum_portion_ceiling + 0.001, maximum_portion_step):
            print actioness_cutoff,
            with open('training_complete_data') as f:

                for line in f:
                    splits = line.strip().split('\t')
                    video_name = splits[0]
                    scores = map(float, splits[1].split(','))
                    video_length = len(scores)
                    this_vid_proposals = []
                    results = create_proposal(scores, actioness_cutoff, maximum_portion)

                    if video_name not in proposal_data['results'].keys():
                        for interval in results.values():
                            if interval['end'] - interval['start'] < 100:
                                continue
                            elif interval['end'] - interval['start'] > 660:
                                continue
                            proposal = {
                                'score': interval['score'],
                                'segment': [interval['start'], interval['end']],
                            }
                            this_vid_proposals += [proposal]
                        proposal_data['results'][video_name] = this_vid_proposals
                        # if '100385400'  in proposal_data['results'].keys():
                        #   if len(proposal_data['results']['100385400']) != 0:
                        #      print 100385400, len(proposal_data['results']['100385400'])
                    else:
                        for interval in results.values():
                            if interval['end'] - interval['start'] < 100:
                                continue
                            elif interval['end'] - interval['start'] > 660:
                                continue
                            # decide whether accept interval or not.
                            flag = 1
                            start_now = int(interval['start'])
                            end_now = int(interval['end'])
                            for pre_proposal in proposal_data['results'][video_name]:
                                start_pre = int(pre_proposal['segment'][0])
                                end_pre = int(pre_proposal['segment'][1])

                                intersection = len(list(set(range(start_pre, end_pre + 1)).intersection(
                                    set(range(start_now, end_now + 1)))))
                                union = len(
                                    list(set(range(start_pre, end_pre + 1)).union(set(range(start_now, end_now + 1)))))
                                # print video_name, intersection/float(union), repeat_ratio
                                if intersection / float(union) > repeat_ratio:
                                    flag = 0
                                    break
                            if flag == 1:
                                proposal = {
                                    'score': interval['score'],
                                    'segment': [interval['start'], interval['end']],
                                }
                                # print proposal
                                proposal_data['results'][video_name] += [proposal]
                                # print proposal_data['results']
    return proposal_data