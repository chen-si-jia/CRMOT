from deep_sort.detection import Detection
from sklearn import preprocessing as sklearn_preprocessing
from application_util import preprocessing
from sklearn.utils.extmath import softmax
from . import linear_assignment
from application_util import visualization
from scipy.optimize import linear_sum_assignment as sklearn_linear_assignment
import cv2
import numpy as np
from opts import opts


RENEW_TIME = 1


class Update:
    def __init__(self, opt, seq, mvtracker, display, view_list):
        self.seq = seq
        self.view_ls = mvtracker.view_ls
        self.tracker = mvtracker
        self.display = display
        self.min_confidence = 0.5
        self.nms_max_overlap = 1
        self.min_detection_height = 0
        self.delta = 0.5
        self.epsilon = 0.5
        self.result = {key: [] for key in self.view_ls}
        self.view_list = view_list
        self.opt = opt

    def create_detections(self, detection_mat, frame_idx, min_height=0):
        if len(detection_mat) == 0:
            return []
        frame_indices = detection_mat[:, 0].astype(np.int)
        mask = frame_indices == frame_idx

        detection_list = []
        for row in detection_mat[mask]:
            bbox, confidence, score_attr, score_text, score_total, feature, id = row[2:6], row[6], row[10], row[11], row[12], row[13:], row[1]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, score_attr, score_text, score_total, feature, id))
        return detection_list

    def select_detection(self, frame_idx, view):
        detections = self.create_detections(
            self.seq[view]["detections"], frame_idx, self.min_detection_height
        )

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections

    def select_view_detection(self, frame_idx, view):
        view_detections = self.create_detections(
            self.seq[view]["view_detections"], frame_idx, self.min_detection_height
        )

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in view_detections])
        scores = np.array([d.confidence for d in view_detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        view_detections = [view_detections[i] for i in indices]
        return view_detections

    def frame_matching(self, frame_idx):
        def gen_X(features):
            features = [sklearn_preprocessing.normalize(i, axis=1) for i in features]
            all_blocks_X = {view: [] for view in self.view_ls}
            for x, view_x in zip(features, self.view_ls):
                row_blocks_X = {view: [] for view in self.view_ls}
                for y, view_y in zip(features, self.view_ls):
                    S12 = np.dot(x, y.transpose(1, 0))
                    scale12 = (
                        np.log(self.delta / (1 - self.delta) * S12.shape[1])
                        / self.epsilon
                    )
                    S12 = softmax(S12 * scale12)
                    S12[S12 < self.opt.cross_view_threshold] = 0
                    assign_ls = sklearn_linear_assignment(-S12)
                    assign_ls = np.asarray(assign_ls)
                    assign_ls = np.transpose(assign_ls) # Transpose
                    X_12 = np.zeros((S12.shape[0], S12.shape[1]))
                    for assign in assign_ls:
                        if S12[assign[0], assign[1]] != 0:
                            X_12[assign[0], assign[1]] = 1
                    row_blocks_X[view_y] = X_12
                all_blocks_X[view_x] = row_blocks_X

            return all_blocks_X

        # print("Matching frame %05d" % frame_idx)
        all_view_features = []
        all_view_id = []
        for view in self.view_ls:
            view_feature = []
            view_id = []
            self.tracker.mvtrack_dict[view].detections = self.select_detection(
                frame_idx, view
            ) # Get the character view features of this frame
            self.tracker.mvtrack_dict[
                view
            ].view_detections = self.select_view_detection(frame_idx, view) # Get the cross-view features of the character in this frame
            for detection in self.tracker.mvtrack_dict[view].view_detections:
                view_feature.append(detection.feature)
                view_id.append(detection.id)
            if view_feature != []:
                view_feature = np.stack(view_feature)
                view_id = np.stack(view_id) # Pack
                view_feature = sklearn_preprocessing.normalize(
                    view_feature, norm="l2", axis=1
                )
                all_view_features.append(view_feature)
            else:
                all_view_features.append(np.array([[0] * self.opt.reid_dim]))
            all_view_id.append(view_id)
        match_mat = gen_X(all_view_features)
        self.tracker.update(match_mat)

    def frame_callback(self, frame_idx):
        if RENEW_TIME:
            re_matching = frame_idx % RENEW_TIME == 0
        else:
            re_matching = 0
        for view in self.view_ls:
            self.tracker.mvtrack_dict[view].predict()
            if view == self.view_list[0]:
                self.tracker.mvtrack_dict[view].pre_update(False)
            else:
                self.tracker.mvtrack_dict[view].pre_update(re_matching)

        for view in self.view_ls:
            linear_assignment.spatial_association(self.tracker, view)
        track_ls = []
        for view in self.view_ls:
            # print(view)
            track_ls += self.tracker.mvtrack_dict[view].matches
            # print("matches:", self.tracker.mvtrack_dict[view].matches)
            track_ls += self.tracker.mvtrack_dict[view].possible_matches
            # print("pos matches:", self.tracker.mvtrack_dict[view].possible_matches)
            track_ls += self.tracker.mvtrack_dict[view].matches_backup
            # print("bu matches:", self.tracker.mvtrack_dict[view].matches_backup)
        track_ls = [i[0] for i in track_ls]
        for view in self.view_ls:
            for track_ in track_ls:
                if track_ in self.tracker.mvtrack_dict[view].unmatched_tracks:
                    self.tracker.mvtrack_dict[view].unmatched_tracks.remove(track_) # Remove from the unmatched list
        for view in self.view_ls:
            self.tracker.mvtrack_dict[view].update()


    def frame_display(self, vis, frame_idx, view):
        # Stores ids corresponding to text
        need_tracks = []
        for track in self.tracker.mvtrack_dict[view].tracks:
            # # ================ Remove PM module ================
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            # # If the score in a single view is greater than the threshold, it will be put into the required tracks.
            # score_total = track.score_total
            # if score_total > self.opt.average_views_total_score_thres:
            #     need_tracks.append(track)

            # ================ PM module ================
            # Set the number of views according to the scene
            sence = self.seq[view]["image_filenames"].split("_")[0]
            if ("Garden1" == sence) or ("ParkingLot" == sence):
                view_list = ["View1", "View2", "View3", "View4"]
            else:
                view_list = ["View1", "View2", "View3"]
            
            id_view_num = 1
            id_views_score_attr = []
            id_views_score_text = []
            id_views_score_total = [] # Total score for each view

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            score_attr = track.score_attr
            score_text = track.score_text
            score_total = track.score_total

            id_views_score_attr.append(score_attr)
            id_views_score_text.append(score_text)
            id_views_score_total.append(score_total)

            id = track.track_id

            for other_view in view_list:
                if view != other_view:
                    for other_view_track in self.tracker.mvtrack_dict[other_view].tracks:
                        if not other_view_track.is_confirmed() or other_view_track.time_since_update > 1:
                            continue
                        other_view_id = other_view_track.track_id
                        if id == other_view_id:
                            score_attr = other_view_track.score_attr
                            score_text = other_view_track.score_text
                            score_total = other_view_track.score_total
                            id_views_score_attr.append(score_attr)
                            id_views_score_text.append(score_text)
                            id_views_score_total.append(score_total)
                            id_view_num += 1
                            break
            
            # Add the id tracks that meet the total score threshold to the list
            # Multi-view average total score threshold:
            if sum(id_views_score_total) / id_view_num > self.opt.average_views_total_score_thres:
                track.score_total_hits += self.opt.hits_1 # The number of hits plus hits_1
                need_tracks.append(track)
            else:
                # The id track of a view greater than the single view total score threshold is added to the list: 
                # (one is very large), and the three views are traversed
                for i, score in enumerate(id_views_score_total):
                    if score > self.opt.single_view_total_score_thres:
                        track.score_total_hits += self.opt.hits_2 # The number of hits plus hits_2
                        for ratio in range(2, 11):
                            if score > ratio * self.opt.single_view_total_score_thres:
                                track.score_total_hits += self.opt.hits_2 # The number of hits plus hits_2
                            else:
                                break
                        if track not in need_tracks:
                            need_tracks.append(track) # Put track directly into need_tracks
                        else:
                            need_tracks.remove(track) # Remove the previous track
                            need_tracks.append(track) # Put the new track in need_tracks
                    else:
                        track.score_total_hits -= self.opt.hits_3 # The number of hits minus hits_3
                        track.score_total_hits = max(track.score_total_hits, 0) # Set track.score_total_hits to a minimum of 0
                # The number of hits scored is greater than a certain threshold
                if track.score_total_hits > self.opt.score_total_hits_thres:
                    # If it is not in need_tracks, then add
                    if track not in need_tracks:
                        need_tracks.append(track)
        if self.opt.test_divo:
            # Update visualization.
            if self.display:
                Path = "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/test/" \
                    + self.seq[view]["image_filenames"].split("_")[0] + "_" + view + "/img1/" \
                    + self.seq[view]["image_filenames"].split("_")[0] + "_" + view + "_" + str(frame_idx).zfill(6) + ".jpg"
                image = cv2.imread(
                    Path,
                    cv2.IMREAD_COLOR,
                )
                # Exception handling. Visualization. Common for testing and training
                if image is None:
                    Path = "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/train/" \
                        + self.seq[view]["image_filenames"].split("_")[0] + "_" + view + "/img1/" \
                        + self.seq[view]["image_filenames"].split("_")[0] + "_" + view + "_" + str(frame_idx).zfill(6) + ".jpg"
                    image = cv2.imread(
                        Path,
                        cv2.IMREAD_COLOR,
                    )
                vis.set_image(image.copy(), view, str(frame_idx))
                vis.draw_trackers(need_tracks)
        if self.opt.test_campus:
            # Update visualization.
            if self.display:
                Path = "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_Cross-domain/images/test/" \
                    + self.seq[view]["image_filenames"].split("_")[0] + "_" + view + "/img1/" \
                    + self.seq[view]["image_filenames"].split("_")[0] + "_" + view + "_" + str(frame_idx).zfill(6) + ".jpg"
                image = cv2.imread(
                    Path,
                    cv2.IMREAD_COLOR,
                )
                # Exception handling. If the image is not read, it will not be processed.
                if image is not None:
                    vis.set_image(image.copy(), view, str(frame_idx))
                    vis.draw_trackers(need_tracks)
                else:
                    print("No image found")
        # Storing inference results
        for track in need_tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            self.result[view].append(
                [frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
            )

    def run(self):
        if self.display:
            visualizer = visualization.Visualization(self.seq, self.opt.exp_name, update_ms=5)
        else:
            visualizer = visualization.NoVisualization(self.seq)
        print("start inference...")
        visualizer.run(self.frame_matching, self.frame_callback, self.frame_display)
