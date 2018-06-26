import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyart
import pdb

############################################
# This program identifies target points from 
# two successive Radar images and calculates
# cloud movement based on matched points.
############################################

def get_radar_data(RADAR_FILE):
	radar = pyart.aux_io.odim_h5.read_odim_h5(RADAR_FILE)
	# mask out last 10 gates of each ray, this removes the "ring" around radar.
	radar.fields['reflectivity']['data'][:, -10:] = np.ma.masked
	# exclude masked gates from the gridding
	gatefilter = pyart.filters.GateFilter(radar)
	gatefilter.exclude_transition()
	gatefilter.exclude_masked('reflectivity')
	# perform Cartesian mapping, limit to the reflectivity field.
	grid = pyart.map.grid_from_radars(
		(radar,), gatefilters=(gatefilter, ),
		grid_shape=(1, 241, 241),
		grid_limits=((2000, 2000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
		fields=['reflectivity'])	
	return grid

def filter_by_speed(kpts1,kpts2,matches,dis_thrld):
    move_dis=[(kpts1[x.queryIdx].pt[0]-kpts2[x.trainIdx].pt[0])**2+(kpts1[x.queryIdx].pt[1]-kpts2[x.trainIdx].pt[1])**2 for (x,y) in matches]
    right_match=[matches[i][0] for i in range(len(move_dis)) if move_dis[i]<dis_thrld]
    movement=[{'pst_start':kpts1[x.queryIdx].pt,'pst_end':kpts2[x.trainIdx].pt} for x in right_match]
    return right_match,movement

def SIFT(img):
    I = cv2.imread(img)
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    kps, features = descriptor.detectAndCompute(gray, None)
    cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Keypoints Detected', I)
    cv2.imwrite('sift_keypoints.jpg',I)
    cv2.waitKey(0)

if __name__ == '__main__':
    # prepare data
    gray1 = get_radar_data('180012318.h5').fields['reflectivity']['data'][0]
    gray2 = get_radar_data('180612318.h5').fields['reflectivity']['data'][0]
    gray1 = (gray1+30.1)/90*255
    gray1[gray1.mask==True]=0
    gray2 = (gray2+30.1)/90*255
    gray2[gray2.mask==True]=0
    img1 =gray1.data.astype(np.uint8)
    img2 =gray2.data.astype(np.uint8)    
    # creat sift and matcher
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})
    # detect keypoints
    kpts1, descs1 = sift.detectAndCompute(gray1.data.astype(np.uint8),None)
    kpts2, descs2 = sift.detectAndCompute(gray2.data.astype(np.uint8),None)
    print('key_pts for 2 images:')
    print(len(kpts1))
    print(len(kpts2))
    # match keypoints
    matches = matcher.knnMatch(descs1, descs2, 2)
    # sort by their distance
    matches = sorted(matches, key = lambda x:x[0].distance)
    print('got matches:')
    print(len(matches))
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
    print('good pts:')
    print(len(good))
    # draw matched pts
    canvas = img2.copy()
    MIN_MATCH_COUNT = 10
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))
    right_match,movement=filter_by_speed(kpts1,kpts2,matches,8**2)
    matched_right=cv2.drawMatches(img1,kpts1,canvas,kpts2,right_match,None)
    cv2.imwrite("matched_pts.png", matched_right)
    # draw target_points
    x=movement[0]
    img1_resized=cv2.resize(img1,(9*241,9*241),interpolation = cv2.INTER_NEAREST)
    img1_resized_rgb=cv2.cvtColor(img1_resized,  cv2.COLOR_GRAY2BGR)
    img2_resized=cv2.resize(img2,(9*241,9*241),interpolation = cv2.INTER_NEAREST)
    img2_resized_rgb=cv2.cvtColor(img2_resized,  cv2.COLOR_GRAY2BGR)    
    # draw area scale of pts
    #img_keyp=cv2.drawKeypoints(img1,[kpts1[x.queryIdx] for x in right_match],img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite("img_keyp.png", img_keyp)
    target_point1=cv2.circle(img1_resized_rgb,(round(x['pst_start'][0]*9+4),round(x['pst_start'][1]*9+4)),3,[0,0,255], thickness=-1)
    target_point2=cv2.circle(img2_resized_rgb,(round(x['pst_end'][0]*9+4),round(x['pst_end'][1]*9+4)),3,[0,0,255], thickness=-1)
    # draw velocity vector
    #speed_arrow=cv2.arrowedLine(img1_resized_rgb, (round(x['pst_start'][0]*9+4),round(x['pst_start'][1]*9+4)), (round(x['pst_end'][0]*9+4),round(x['pst_end'][1]*9+4)), [0,0,255], thickness=3)
    for x in movement:
        target_point1=cv2.circle(target_point1,(round(x['pst_start'][0]*9+4),round(x['pst_start'][1]*9+4)),3,[0,0,255], thickness=-1)
        target_point2=cv2.circle(target_point2,(round(x['pst_end'][0]*9+4),round(x['pst_end'][1]*9+4)),3,[0,0,255], thickness=-1)
        #speed_arrow=cv2.arrowedLine(speed_arrow, (round(x['pst_start'][0]*9+4),round(x['pst_start'][1]*9+4)), (round(x['pst_end'][0]*9+4),round(x['pst_end'][1]*9+4)), [0,0,255], thickness=3)   #, thickness=1, line_type=8, shift=0, tipLength=0.1
    cv2.imwrite("target_point1.png", target_point1)
    cv2.imwrite("target_point2.png", target_point2)
    #cv2.imwrite("speed_arrow.png", speed_arrow)