# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from PIL.Image import new
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from deep_sort import track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance_x metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance_x metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, max_iou_distance=0.7, max_age=60, n_init=3):
    # def __init__(self, metric, max_iou_distance=0.7, max_age=60, n_init=3):
    #     self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.state = track.TrackState
        self.tracks = []
        self.newbie_tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, width, frame_num, frame_skip):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed(width)
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], frame_num)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        newbie_idx = [i for i,t in enumerate(self.tracks) if t.track_id in self.newbie_tracks and t.age < 200]
        occluded_idx = [i for i,t in enumerate(self.tracks) if t.is_occluded()]

        if frame_num % frame_skip == 0 and len(occluded_idx) != 0 and len(newbie_idx) != 0:

            # array for newbie
            candidates_x = np.array([]) 
            for  n in newbie_idx:
                candidates_x = np.append(candidates_x ,self.tracks[n].mean[0])

            remove_idx, new_matching = [], []

            for tracks in occluded_idx:
                if len(candidates_x) == 0:
                    break
                if tracks in remove_idx:
                    continue
                # calculate distance_x : occluded - new
                distance_x = abs(self.tracks[tracks].mean[0] - candidates_x[:])

                if len(distance_x) == 1 and distance_x == 0:
                    continue
                minval_x = np.min(distance_x[np.nonzero(distance_x)])
                idx = np.where(distance_x == minval_x)[0][0]

                # if height of the new box is small
                if self.tracks[newbie_idx[idx]].mean[3] < 70:
                    gap = 65
                else:
                    gap = 105
                if minval_x < gap  :
                    # idx in newbie with nearest distance_x
                    if abs( self.tracks[tracks].mean[1] - self.tracks[newbie_idx[idx]].mean[1] ) > 35 :
                        continue

                    if newbie_idx[idx] in remove_idx:
                        # get the location in new matching
                        # location = self.index_2d(new_matching, newbie_idx[idx])
                        for i, x in enumerate(new_matching):
                            if newbie_idx[idx] in x:
                                location = i                        
                        # compare which is smaller
                        # if current is smaller, remove older list
                        if minval_x < new_matching[location][2] :
                            del new_matching[location]
                            del remove_idx[location]
                        else:
                            continue
                    new_matching.append((tracks, newbie_idx[idx], minval_x))
                    # save this n into list for remove
                    remove_idx.append(newbie_idx[idx])

            # update/change newbie track with old one
            for old, new, _ in new_matching:
                self.tracks[old].re_id(self.kf, self.tracks[new].mean[:4])                
            # remove track from newbie list
            newbie_idx = [i for  i in newbie_idx if i not in remove_idx]
            # delete this new track id from self.tracks
            self.tracks = [i for j, i in enumerate(self.tracks) if j not in remove_idx]

    def _match(self, detections):
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates =  [  i for i, t in enumerate(self.tracks)]
       
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates)

        matches = matches_b
        unmatched_tracks = list(set(unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, frame_num):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            class_name))

        if frame_num != 1: 
            self.newbie_tracks.append( self._next_id)
        self._next_id += 1
