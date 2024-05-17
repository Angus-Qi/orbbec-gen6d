import argparse
import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from prepare import video2image
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp

import cv2
import time
from pyorbbecsdk import *
from utils.utils import frame_to_bgr_image

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts

def get_pose(img, pose_init, hist_pts, estimator, ref_database):
    
    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d, midpt3d, = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))
    h, w, _ = img.shape
    f=np.sqrt(h**2+w**2)
    K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32) #use own camera intrinsics

    if pose_init is not None:
        estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
    pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init)
    pose_init = pose_pr

    pts, _ = project_points(object_bbox_3d, pose_pr, K) #pts is 2d    
    hist_pts.append(pts)
    weight_num = 5
    arg_std = 2.5
    pts_ = weighted_pts(hist_pts, weight_num = weight_num, std_inv= arg_std)
    pose_, dist_coeff = pnp(object_bbox_3d, pts_, K) #object_bbox_3d is 3d, pts_ is 2d
    pts__, _ = project_points(object_bbox_3d, pose_, K) # pose_ is [R;t]
    bbox_img_ = draw_bbox_3d(img, pts__, (0,0,255)) #pts__ is 2d

    # for ctrpoint
    midpt2d, _ = project_points(midpt3d, pose_, K)
    #pnp func need minimum 4 points 

    return bbox_img_, pose_init, pose_, K, dist_coeff, midpt2d

def main(args):
    #some variables
    pressedKey = None
    pause = False
    frameCount = 0
    startTime = None
    frameFpsText = None
    fpsBgColor = (0, 0, 0)
    fpsColor = (255, 255, 255)
    fpsFontType = cv2.FONT_HERSHEY_SIMPLEX
    fpsLineType = cv2.LINE_AA

    pose_init = None
    hist_pts = []
    cfg = load_cfg(args.cfg)
    ref_database = parse_database_name(args.database)
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')

    #Open camera
    config = Config()
    pipeline = Pipeline()
    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline.enable_frame_sync()
    temporal_filter = TemporalFilter(alpha=0.8)

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30) #640x360, 30fps, somehow higher fps reduce latency
    config.enable_stream(color_profile)
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.start(config)

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(0)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
        except KeyboardInterrupt:
            break
        color_image = frame_to_bgr_image(color_frame)
        
        depth_frame = frames.get_depth_frame()
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        depth_data = temporal_filter.process(depth_data)
        
        # #Show Depth map
        # depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # depth_image = (depth_data * (255 / 5000)).astype(np.uint8) #Normalisation, max depth is 10000mm
        # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        # color_image = cv2.resize(color_image, (0, 0), fx=1/2, fy=1/2)
        # disparityFrame = cv2.resize(disparityFrame, (0, 0), fx=1/3, fy=1/3)
        
        frameCount += 1
        if frameCount%20==0: # Recompute every 20 frames
            
            if startTime:
                timeNow = time.time() #Return time in seconds, float
                frameFps = round(20 / (timeNow - startTime), 2)
                frameFpsText = str(frameFps)

            pose_init = None
            startTime = time.time()
        
        pose_im, pose_init, sixDPose, camera_matrix, dist_coefficient, midpt2d = get_pose(color_image, pose_init, hist_pts, estimator, ref_database)
        
        ### Show FPS ###
        if frameFpsText:
            cv2.putText(pose_im, frameFpsText, (5, 15), fpsFontType, 0.5, fpsBgColor, 4, fpsLineType)
            cv2.putText(pose_im, frameFpsText, (5, 15), fpsFontType, 0.5, fpsColor, 1, fpsLineType)
        ##################
        
        #### Draw pose axis #######
        """
            Draw the 6DoF pose axis on an image using the rotation matrix and translation vector.
            
            Parameters:
                rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
                translation_vector (numpy.ndarray): 3x1 translation vector.
                camera_matrix (numpy.ndarray): 3x3 camera matrix.
                dist_coefficients (numpy.ndarray): Distortion coefficients.
                scale (float): Length of the axis lines.

        """
        rotation_matrix = np.array(sixDPose[:3, :3], dtype=np.float64) #top-left 3x3 submatrix
        translation_vector = np.array(sixDPose[:,3], dtype=np.float64)
        camera_matrix = np.array(camera_matrix, dtype=np.float64)
        dist_coefficient = np.array(dist_coefficient, dtype=np.float64)
        ############################

        midpt2d = tuple(np.int32(midpt2d)[0]) # x,y
        print('midpt2d: ', midpt2d)

        try: # Index range check
            ctrDisparity = depth_data[midpt2d[1]][midpt2d[0]] 
        except:
            print("Midpoint estimated out of frame")
        print('depth at ctr point: %.2fmm' % ctrDisparity) 

        # 6d metric pose, d**2 = x**2 + y**2 + z**2 by pythagoras theorem, frame centre is [0,0,0]
        #Right is positive x, up is negative y, inside is positive z
        #Diameter of custom database object set to 2.0, therefore minus 1.0 from diagonal distance to account for obj radius
        camera_space = translation_vector * ctrDisparity/(((translation_vector[-1])**2+translation_vector[-2]**2+translation_vector[-3]**2)**0.5 - 1.0)
        print('Metric location of object in camera space: ', camera_space)

        #Draw obj centre
        cv2.circle(pose_im, midpt2d, radius=10, color=(255, 127, 127),thickness= -1)
        #Draw axis lines
        scale = 1.0
        axis_points = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(axis_points, rotation_matrix, translation_vector, camera_matrix, dist_coefficient) #input matrix should be in np.float64
        image_points = np.int32(image_points).reshape(-1, 2)
        xDiff = image_points[1]-image_points[0]
        yDiff = image_points[2]-image_points[0]
        zDiff = image_points[3]-image_points[0]
        cv2.line(pose_im, midpt2d, midpt2d+xDiff, (0, 0, 255), 2)  # X-axis (Red)
        cv2.line(pose_im, midpt2d, midpt2d+yDiff, (0, 255, 0), 2)  # Y-axis (Green)
        cv2.line(pose_im, midpt2d, midpt2d+zDiff, (255, 0, 0), 2)  # Z-axis (Blue)
        cv2.imshow("Colour image", pose_im)

        # # Show on depth map
        # cv2.circle(depth_image,midpt2d, radius=10, color=(255, 127, 127),thickness= -1)
        # cv2.line(pose_im, tuple(image_points[0]), tuple(image_points[1]), (0, 0, 255), 2)  # X-axis (Red)
        # cv2.line(pose_im, tuple(image_points[0]), tuple(image_points[2]), (0, 255, 0), 2)  # Y-axis (Green)
        # cv2.line(pose_im, tuple(image_points[0]), tuple(image_points[3]), (255, 0, 0), 2)  # Z-axis (Blue)
        # cv2.imshow("Depth map", depth_image)

        ## Key Control ###########################
        if pressedKey == None:
            pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('p'): #Pause
            pause = True
        elif pressedKey == ord('q'): #Quit
            break
        while pause:
            time.sleep(0.5)
            if cv2.waitKey(1) == ord('c'): #Continue
                pause = False
        pressedKey = None
        ##########################################

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str, default="custom/mouse")
    args = parser.parse_args()
    
    main(args)