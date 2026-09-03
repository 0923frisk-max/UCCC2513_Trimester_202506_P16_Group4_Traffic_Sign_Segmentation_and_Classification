#Code Sample 4 [Done by: Lee Jun Xian]
import cv2
import numpy as np
import os
import glob
import math


def get_hsv_ranges(color_name):
    if color_name == "Blue":
        return [(np.array([90, 60, 20]), np.array([140, 255, 255]))]
   
    elif color_name == "Yellow":
        return [(np.array([12, 50, 50]), np.array([32, 255, 255]))]
   
    elif color_name == "Red":
        lower_red1 = np.array([0, 40, 30])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 40, 30])
        upper_red2 = np.array([180, 255, 255])
        return [(lower_red1, upper_red1), (lower_red2, upper_red2)]
   
    else:
        return []


def batch_process_signs(input_folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    total_saved = 0


    for folder in input_folders:
        color_target = ""
        if "Blue" in folder: color_target = "Blue"
        elif "Red" in folder: color_target = "Red"
        elif "Yellow" in folder: color_target = "Yellow"
        else: continue


        image_paths = glob.glob(os.path.join(folder, "*.*"))
        if not image_paths:
            continue


        print(f"Processing '{folder}' ({color_target} )...")


        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
               
            img = cv2.resize(img, (300, 300))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


            ranges = get_hsv_ranges(color_target)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, current_mask)


           
            kernel_open = np.ones((5, 5), np.uint8)
            kernel_close = np.ones((11, 11), np.uint8)
           
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)


           
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(mask)


            valid_contours = []
           
            if contours:
                for c in contours:
                    area = cv2.contourArea(c)
                   
                   
                    if 500 < area < 80000:
                        x, y, w, h = cv2.boundingRect(c)
                        aspect_ratio = float(w) / float(h)
                       
                       
                        if 0.4 <= aspect_ratio <= 2.5:
                           
                            hull = cv2.convexHull(c)
                            hull_area = cv2.contourArea(hull)
                            if hull_area > 0:
                                solidity = float(area) / hull_area
                               
                                if solidity > 0.25:
                                    valid_contours.append(c)


           
            if valid_contours:
               
                best_contour = max(valid_contours, key=cv2.contourArea)
               
               
                epsilon = 0.035 * cv2.arcLength(best_contour, True)
                approx = cv2.approxPolyDP(best_contour, epsilon, True)
                vertices = len(approx)


               
                if vertices >= 6:
                   
                    M = cv2.moments(best_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(best_contour)
                        if area > 0:
                           
                            radius = int(math.sqrt(area / math.pi) * 1.02)
                            cv2.circle(final_mask, (cX, cY), radius, 255, thickness=cv2.FILLED)
                else:
                   
                   
                    cv2.drawContours(final_mask, [approx], -1, 255, thickness=cv2.FILLED)


           
            segmented_sign = cv2.bitwise_and(img, img, mask=final_mask)


           
            comparison_display = np.hstack((img, segmented_sign))
            cv2.putText(comparison_display, 'Original', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_display, 'Sign segmented', (320, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


           
            base_filename = os.path.basename(img_path)
            save_path = os.path.join(output_folder, f"result_{color_target}_{base_filename}")
            cv2.imwrite(save_path, comparison_display)
            total_saved += 1


    print(f"✅ Process success total {total_saved} picture save in '{output_folder}' folder。")


if __name__ == "__main__":
    FOLDERS_TO_PROCESS = ["Blue Signs", "Red signs", "Yellow Signs"]
    OUTPUT_DIRECTORY = "final_enhanced_results"
    batch_process_signs(FOLDERS_TO_PROCESS, OUTPUT_DIRECTORY)

