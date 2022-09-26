import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import math

# Default coordinates for outer rectangle:
x_min=1
x_max=459
y_min=50
y_max=299

# Default coordinates and radius for search area

center = (209,139)
radius = 68


def contour_in_roi(contours, i, center, radius):
    ''' Function to determine if area is in ROI
    '''
    
    cx, cy = center
    
    for j in range(len(contours[i])):
        x = contours[i][j][0][0]
        y = contours[i][j][0][1]
        
        dist = math.sqrt((x-cx)**2 + (y-cy)**2)
        
        if dist <= radius:
            return True
    
    return False

def all_contour_in_roi(contours,i, x_min, x_max, y_min, y_max):
    ''' Function to determine if area is completely within ROI
    '''

    for j in range(len(contours[i])):
        x = contours[i][j][0][0]
        y = contours[i][j][0][1]
        
        if (x_min < x < x_max) & (y_min < y < y_max):
            point_in_roi = True
        else:
            return False
            
    return point_in_roi

def bs_detect(img, x_min, x_max, y_min, y_max, select_punched = False):
    '''
    Detect blood spots in an image using threshold and contours
    
    Internal contours that lie completely within rectangle bounded by (x_min,y_min) to (x_max,y_max) are filled
    
    If select_punched = True then only blood spots with green punch annotations are selected
    
    img --> contours, hierarchy
    
    ''' 
    
    
    ## basic blood spot algorithm
    
    blurred_img = cv2.medianBlur(img,3)      
    gray = cv2.cvtColor(blurred_img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # add background to remove enclosing artefacts
    h, w = thresh.shape
    background = np.zeros((h,w),dtype=np.uint8)
    x = int(w/2)
    rad = int(w/2)
    y = int(h/2)
    background = cv2.circle(background,(x,y),rad,255,-1)
    thresh = cv2.bitwise_and(background,thresh)
    
    # foreground noise reduction
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # find internal contours without noise reduction
    s_contour_image, s_contours, s_hierarchy = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # fill internal contours on thresholded image
    for i in range(len(s_contours)):
        # skip if contour is outside region of interest
        if not all_contour_in_roi(s_contours,i,x_min,x_max,y_min,y_max):
            continue
        ## last field in heirarchy is -1 if the contour has no parent - so select only those with parent
        if s_hierarchy[0][i][3] != -1: 
            # fill contours on threshold image, only if contour is large enough to be a punch
            if cv2.contourArea(s_contours[i]) > 100:
                cv2.drawContours(closing, s_contours, i, (255,255,255),thickness=cv2.FILLED)
        
    # noise removal on amended thresholded image
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = 5)
    
    # find contours --> blood spots
    contour_image, contours, hierarchy = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if not select_punched:
        return contours, hierarchy
    
    elif select_punched:
        
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        for i in range(len(contours)):
            # fill external contours
            cv2.drawContours(contour_image, contours, i, (255,255,255), thickness=cv2.FILLED)               

        # blur image for green mask - use a separate blur in case the optimum kernal size differs    
        blurred_img_gm = cv2.medianBlur(img_rgb,7)

        ## convert to hsv and mask of green (36,25,25) ~ (86, 255,255)
        hsv = cv2.cvtColor(blurred_img_gm, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))

        # mask erosion
        kernel = np.ones((9,9),np.uint8)
        mask_erode = cv2.erode(mask,kernel,iterations = 2)
        mask_erode = cv2.medianBlur(mask_erode,3)

        # add mask to grayscale image (so that punches appear as white)
        add_punch = cv2.subtract(contour_image,mask_erode)

        ## detect contours of punched spot
        p_contour_image, p_contours, p_hierarchy = cv2.findContours(add_punch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        return p_contours, p_hierarchy


def draw_bs_contours(img,contours,hierarchy, center, radius, label_contours=False, select_punched=False, punch_mode='center'):
    '''
    draw all detected blood spot contours in circular region of interest

    If select_punched = True then only blood spots with green punch annotations are selected,
    and the location of punch contours is indicated
    
    If label_contours = True then contours are labelled with a contour number
    
    Punch mode:
        'outline' - draw the outline of the punch contour
        'center' - draw a point at the center of the punch contour
    
    '''
    
    # define critera for selecting blood spots based on image type
    if select_punched:
        # third column in the array is -1 if it does not have a child contour
        criteria = 'hierarchy[0][i][2] != -1'
    else:
        # last column in the array is -1 if an external contour (no contours inside of it)
        criteria = 'hierarchy[0][i][3] == -1'
    
    for i in range(len(contours)):
        
        # skip if contour is outside region of interest
        if not contour_in_roi(contours,i,center,radius):
            continue
                
        # last column in the array is -1 if an external contour (no contours inside of it)
        if eval(criteria): 
            # Draw the external contours from the list of contours
            cv2.drawContours(img, contours, i, (0,0,255), 2)
            
            if label_contours:
                M=cv2.moments(contours[i])
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                cv2.putText(img,text=str(i),org=(cx-60,cy-60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=2,lineType=cv2.LINE_AA)

            # Draw the punch contours
            if select_punched:
                for j in range(len(contours)):
                    if hierarchy[0][j][3] == i:
                        
                        if punch_mode == 'outline':
                            cv2.drawContours(img, contours, j, (255,0,0), 2)
                        
                        elif punch_mode == 'center':
                            # Calculate the co-ordinates of the center of the punch
                            M_punch = cv2.moments(contours[j])
                            punch_cX = int(M_punch["m10"] / M_punch["m00"])
                            punch_cY = int(M_punch["m01"] / M_punch["m00"])

                            # Draw the center of the punch on the image
                            cv2.circle(img, (punch_cX, punch_cY), 5, (255, 0, 0), -1)
                            
                        else:
                            raise Exception("Punch mode must be equal to 'outline' or 'center'")
                        
    return img

def spot_metrics(contours,hierarchy,mm_per_pixel, center, radius, select_punched = False):
    
    spot_list = []
    hull = []


    # define critera for selecting blood spots based on image type
    if select_punched:
        # third column in the array is -1 if it does not have a child contour
        criteria = 'hierarchy[0][i][2] != -1'
    else:
        # last column in the array is -1 if an external contour (no contours inside of it)
        criteria = 'hierarchy[0][i][3] == -1'
    
    for i in range(len(contours)):
        
        hull.append(cv2.convexHull(contours[i], False))
        
        if not contour_in_roi(contours,i,center,radius):
            continue
                
        # last column in the array is -1 if an external contour (no contours inside of it)
        if eval(criteria):
            number_punches = 0
            punch_area_list = []
            punch_distance_list = []
            average_punch_area=False
            average_punch_mm_dist_from_center=False
            average_punch_dist_from_center_prop=False
            
            ## BLOOD SPOT METRICS
            
            area = cv2.contourArea(contours[i])
            perimeter = cv2.arcLength(contours[i],True)
            perimeter_mm = perimeter*mm_per_pixel

            # calculate circumference and diameter of circle with area the same size as contour            
            equiv_diam = np.sqrt(4*area/np.pi)
            equiv_diam_mm = equiv_diam*mm_per_pixel
            equiv_circ = np.sqrt(4*np.pi*area)

            # calculate ratio of circumference of theoretical circle to perimeter
            # values close to 1 indicate a round bloodspot. values close to 0 indicate very irregular blood spot
            roundness = equiv_circ/perimeter

            # calculate minimum area bounding rectangle
            (x,y),(w,h),angle = cv2.minAreaRect(contours[i])

            if w>h:
                long=w
                short=h
            else:
                long=h
                short=w
            long_mm=long*mm_per_pixel
            short_mm=short*mm_per_pixel
            elongation=short/long

            # calculate minimum enclosing circle
            (x,y),min_circ_rad = cv2.minEnclosingCircle(contours[i])
            min_circle_area = np.pi*min_circ_rad*min_circ_rad
            circular_extent=area/min_circle_area

            # convex hull properties
            hull_area = cv2.contourArea(hull[i])
            solidity = area/hull_area
            hull_perimeter = cv2.arcLength(hull[i],True)
            convexity = hull_perimeter/perimeter

            # Calculate the co-ordinates of the center of the blood spot
        
            M_spot = cv2.moments(contours[i])
            spot_cX = int(M_spot["m10"] / M_spot["m00"])
            spot_cY = int(M_spot["m01"] / M_spot["m00"])
            
            ### PUNCH METRICS
            
            if select_punched:
                # find child contours (punches) of blood spot
                for j in range(len(contours)):
                    if hierarchy[0][j][3] == i:
                        # count number of punches for each spot
                        number_punches += 1

                        # calculate area of each punch and append to a list
                        punch_area = cv2.contourArea(contours[j])
                        punch_area_list.append(punch_area)

                        # Calculate the co-ordinates of the center of the punch
                        M_punch = cv2.moments(contours[j])
                        punch_cX = int(M_punch["m10"] / M_punch["m00"])
                        punch_cY = int(M_punch["m01"] / M_punch["m00"])

                        # Calculate the distance between the center of the blood spot and punch and append to list
                        punch_pixel_dist_from_center = math.sqrt((punch_cX-spot_cX)**2 + (punch_cY-spot_cY)**2)
                        punch_distance_list.append(punch_pixel_dist_from_center)

                        # calculate averages
                        average_punch_area = sum(punch_area_list)/len(punch_area_list)       
                        average_punch_pixel_dist_from_center = sum(punch_distance_list)/len(punch_distance_list)                                
                        average_punch_mm_dist_from_center = average_punch_pixel_dist_from_center*mm_per_pixel
                        
                        average_punch_dist_from_center_prop = average_punch_mm_dist_from_center/equiv_diam_mm
                
            # add area of blood spot, number of punches and average punch area to list
            spot_list.append([i,area,perimeter_mm,roundness, equiv_diam_mm, long_mm,short_mm,
                            elongation, circular_extent, hull_area, solidity, hull_perimeter, convexity,
                              number_punches, average_punch_area,average_punch_mm_dist_from_center,
                             average_punch_dist_from_center_prop])

    return spot_list

def spot_metrics_multi(input_folder,x_min,x_max,y_min,y_max,mm_per_pix,center,radius,image_type,select_punched=False):
    '''
    Calculate metrics on multiple images
    
    Image type:
    'screenshot' - a screenshot from the Panthera PC
    'original' - the original view from the Panthera camera
    'cropped' - an image which has been cropped to within the gripping hand
    
    '''
    spot_list = []
    cols = ['file','contour_index','area','perimeter_mm','roundness','equiv_diam_mm', 
            'long_mm','short_mm', 'elongation', 'circular_extent',
             'hull_area', 'solidity', 'hull_perimeter', 'convexity', 
            'number_punches', 'average_punch_area',
            'average_punch_dist_from_center_mm','average_punch_dist_from_center_prop']
    
    for file in os.listdir(input_folder):

        # skip non-image files
        if not (file.endswith('.jpg') or file.endswith('png')):
            continue

        # read image
        img = cv2.imread(input_folder + "/"+ file)
        
        if image_type == 'screenshot':
            img = img[230:530, 509:969]

        elif image_type =='original':
            img = img[0:300, 150:610]
        
        elif image_type == 'cropped':
            pass
        
        else:
            raise Exception("Image type must be either 'screenshot', 'original' or 'cropped'")
        
        #if image is already cropped then no further crop is necessary
                 
        # detect punched blood spot
        contours, hierarchy = bs_detect(img,x_min,x_max,y_min,y_max,select_punched=select_punched)

        # calculate blood spot metrics
        spot_met = spot_metrics(contours,hierarchy,mm_per_pix,center,radius,select_punched=select_punched)

        for row in spot_met:
            row.insert(0,file[:-4])

        spot_list.extend(spot_met)

    df = pd.DataFrame(spot_list, columns=cols)
    
    return df


def calc_multispot_prob(spot_metrics,columns,ml_columns,scaler,model,scale=True):
    '''
    return multispot probability from spot metrics (list of lists)
    
    '''
    
    ms_prob_list = []
    
    for cnt in spot_metrics:
    # punched spot metrics returns a list within a list

        values = cnt
        dictionary = dict(zip(columns,values))
        contour_index = dictionary['contour_index']
        ml_values = []

        for item in ml_columns:
            ml_values.append(dictionary[item])

        # reshape X (necessary for machine learning model)
        X = [ml_values]

        # scale
        if scale:
            scaled_X = scaler.transform(X)
        else:
            scaled_X = X

        # predict probability
        prob_multi = round(model.predict_proba(scaled_X)[0][0],4)

        ms_prob_list.append((contour_index, prob_multi))
    
    return ms_prob_list

def calc_multispot_prob_multi(spot_metrics_df,ml_columns,scaler,model,scale=True):
    '''
    adds columns for multispot prediction and probability from spot metrics dataframe
    
    '''
    
    X = spot_metrics_df[ml_columns]
    
    if scale:
        scaled_X = scaler.transform(X)
    else:
        scaled_X = X
    
    # calculate predictions and convert to dataframe
    pred_multi = model.predict(scaled_X)
    pred_multi_df = pd.DataFrame(pred_multi,columns=['pred_multi'])

    prob_multi = model.predict_proba(scaled_X)
    prob_multi_df = pd.DataFrame(prob_multi,columns=['prob_multi','prob_control'])

    joined_multi = pred_multi_df.join(prob_multi_df)
    joined_df = spot_metrics_df.reset_index(drop=True).join(joined_multi)
    
    return joined_df

def bs_crop(img, contours,hierarchy, center, radius, select_punched = False, border=5):
    '''
    Returns bounding rectangle for contour
    '''
    
    bounding_box_list = []
    
    # define critera for selecting blood spots based on image type
    if select_punched:
        # third column in the array is -1 if it does not have a child contour
        criteria = 'hierarchy[0][i][2] != -1'
    else:
        # last column in the array is -1 if an external contour (no contours inside of it)
        criteria = 'hierarchy[0][i][3] == -1'
    
    for i in range(len(contours)):
        
        if not contour_in_roi(contours,i,center,radius):
            continue
                
        # last column in the array is -1 if an external contour (no contours inside of it)
        if eval(criteria):

            # calculate bounding rectangle
            x,y,w,h = cv2.boundingRect(contours[i])
            
            # crop to bounding box (if possible)
            
            try:
                cropped_image = img[y-border:y+h+border, x-border:x+w+border]
                reshaped_image = cv2.resize(src=cropped_image,dsize=(100,100))
                bounding_box_list.append([i, reshaped_image])
                
            except:
                cropped_image = img[y:y+h, x:x+w]
                reshaped_image = cv2.resize(src=cropped_image,dsize=(100,100))
                bounding_box_list.append([i, reshaped_image])            
            
    return bounding_box_list

def bs_multicrop(input_folder, output_folder, image_type, x_min,x_max,y_min,y_max,
                 center, radius, select_punched = False, border=5):
    '''
    Crop blood spots on multiple images
    
    Image type:
    'screenshot' - a screenshot from the Panthera PC
    'original' - the original view from the Panthera camera
    'cropped' - an image which has been cropped to within the gripping hand
    
    '''
    
    for file in os.listdir(input_folder):

        # skip non-image files
        if not (file.endswith('.jpg') or file.endswith('png')):
            continue
        
        # read image
        img = cv2.imread(input_folder + "/"+ file)

        if image_type == 'screenshot':
            img = img[230:530, 509:969]

        elif image_type =='original':
            img = img[0:300, 150:610]

        elif image_type == 'cropped':
            pass

        else:
            raise Exception("Image type must be either 'screenshot', 'original' or 'cropped'")

        # detect punched blood spot
        contours, hierarchy = bs_detect(img,x_min,x_max,y_min,y_max,select_punched=select_punched)

        # crop blood spot
        cropped_img_list = bs_crop(img, contours, hierarchy, center, radius, select_punched=select_punched)
        
        # save cropped image
        for img in cropped_img_list:
            
            if len(cropped_img_list) == 1:
                filename = output_folder + "/" + file[:-4] + ".jpg"
            else:
                filename = output_folder + "/" + file[:-4] + "-" + str(img[0]) + ".jpg"
                print('MORE THAN ONE CONTOUR FROM THIS IMAGE')
                
                
            print('Saving ' + file[:-4] + ' contour ' + str(img[0]))
        
            cv2.imwrite(filename, img[1])

def draw_bounding_box(img,contours,hierarchy, center, radius, spot_metrics, multispot_prob_list, select_punched, 
                      diam_range = (8,14), prob_multi_limit = 0.50, prob_multi_borderline = 0.25):
    '''
    Draw colour coded bounding box around blood spots on the 'img'
    '''
    
    # define critera for selecting blood spots based on image type
    if select_punched:
        # third column in the array is -1 if it does not have a child contour
        criteria = 'hierarchy[0][i][2] != -1'
    else:
        # last column in the array is -1 if an external contour (no contours inside of it)
        criteria = 'hierarchy[0][i][3] == -1'
        
    
    # loop through all contours to select contour of interest (i.e. blood spot)
    
    for i in range(len(contours)):
    
        # skip if contour is outside region of interest
        if not contour_in_roi(contours,i,center,radius):
            continue
        
        # use criteria to select correct blood spots based on image type
        if eval(criteria):
            
            #find input variables
            for cnt in spot_metrics:
                if cnt[0] == i:
                    diameter = cnt[4]

            for cnt in multispot_prob_list:
                if cnt[0] == i:
                    prob_multi = cnt[1]
            
            # set outputs
            small=False
            large=False
            multispotted=False
        
            # use parameters to determine if blood spot is unsuitable
            
            # small
            if diameter < diam_range[0]:
                small=True
            
            if diameter > diam_range[1]:
                large=True
            
            if prob_multi >= prob_multi_limit:
                multispotted='+'
            elif prob_multi >= prob_multi_borderline:
                multispotted='b'  
            else:
                multispotted='0' 
            
            # define colour for bounding boxes
            box_colour = (0,255,0)
            diam_colour = (0,255,0)
            multiprob_colour = (0,255,0)
            
            if small or large:
                diam_colour = (255,0,0)

            if multispotted == '+':
                multiprob_colour = (255,0,0)
                
            if multispotted == 'b':
                multiprob_colour = (255,191,0) 
                box_colour = (255,191,0)
                
            if small or large or (multispotted == '+'):
                box_colour = (255,0,0)
            
            # draw bounding box
            x,y,w,h = cv2.boundingRect(contours[i])
            img = cv2.rectangle(img,(x,y),(x+w,y+h),box_colour,2)

            # write blood spot diameter
            img = cv2.putText(img, text=str(round(diameter,1)), org=(x-25,y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=1, color=diam_colour, thickness=2, lineType=cv2.LINE_AA)  

            img = cv2.putText(img, text=str(round(prob_multi,3)), org=(x+w-25,y+h+25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=1, color=multiprob_colour, thickness=2, lineType=cv2.LINE_AA)  
            
    return img

def image_multi_plot(df,rows,columns,title,figsize,select_punched,model,bounding_box=True, diam_range = (7.5,15.5), prob_multi_limit = 0.50):
    fig = plt.figure(figsize=figsize,dpi=200)
    plt.suptitle(title)

    files = df['file'].values.tolist()
    i = 1

    for file in files:

        img = cv2.imread('images/cropped/' + str(file) + '.jpg')
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # detect punched spot
        contours, hierarchy = bs_detect(img, x_min, x_max, y_min, y_max, select_punched)
        
        # draw contours
        contour_img = draw_bs_contours(img_rgb.copy(),contours,hierarchy, center, radius, select_punched)
        
        if bounding_box:
            # calculate properties
            spot_met = spot_metrics(contours,hierarchy,mm_per_pixel,center,radius, select_punched)
            prob_ms_list = calc_multispot_prob(spot_met,cols,ml_cols,scaler,model)

            # draw bounding box
            bounding_box_img = draw_bounding_box(contour_img.copy(),contours,hierarchy, center, radius, spot_met,
                                             multispot_prob_list=prob_ms_list, select_punched=select_punched, diam_range=diam_range,
                                     prob_multi_limit=prob_multi_limit)

        fig.add_subplot(rows,columns, i)

        i += 1

        if bounding_box:
            plt.imshow(bounding_box_img)
        else:
            plt.imshow(contour_img)
        plt.axis('off')
        plt.title(file)

    plt.show()