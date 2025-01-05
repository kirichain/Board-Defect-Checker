from PySide6.QtCore import QThread, Signal
import cv2  # OpenCV for image processing
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class EarPatternAnalyzer:
    """
    Analyzes contours to detect and score ear patterns.
    Left ear should be Γ shaped
    Right ear should be ⅂ shaped
    """

    def __init__(self):
        self.min_score_threshold = 0.5  # Minimum score to consider as valid ear pattern

    def extract_ear_from_binary_image(self, box, contours, box_index, ear_position, binary_image, show_image=False):
        """
        Extracts an ear object from a binary image based on the given box and contour information.

        Args:
            box (tuple): Box coordinates (x, y, w, h)
            contours (list): List of contours detected in the binary image
            box_index (int): Index of the contour/box to process
            ear_position (str): Position identifier ("left" or "right")
            binary_image (ndarray): Input binary image (grayscale or BGR)
            show_image (bool): If True, displays the extracted object

        Returns:
            ndarray: Extracted ear object as a binary image
        """
        # Check if image is BGR and convert to grayscale if needed
        if len(binary_image.shape) == 3:
            print("Converting BGR image to grayscale")
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        elif len(binary_image.shape) != 2:
            print("Error: Input image must be either grayscale or BGR")
            return None

        # Create a mask of zeros with same size as input image
        mask = np.zeros(binary_image.shape, dtype=np.uint8)

        # Get the contour corresponding to the box_index
        if 0 <= box_index < len(contours):
            target_contour = contours[box_index]

            # Draw the contour on the mask in white
            cv2.drawContours(mask, [target_contour], -1, (255, 255, 255), -1)

            # Extract box coordinates
            x, y, w, h = box

            # Create a region of interest (ROI) using the box coordinates
            roi_mask = mask[y:y + h, x:x + w]
            roi_binary = binary_image[y:y + h, x:x + w]

            # Apply the mask to get only the object
            extracted_object = cv2.bitwise_and(roi_binary, roi_binary, mask=roi_mask)

            if show_image:
                window_title = f"Extracted {ear_position} ear object (Box {box_index})"
                cv2.imshow(window_title, roi_binary)
                cv2.waitKey(0)

            return roi_binary
        else:
            print(f"Error: Box index {box_index} is out of range for contours list")
            return None

    def filter_boxes(self, binary_image, boxes, contours, ear_position):
        """
        Filters boxes based on basic criteria before detailed analysis.

        Args:
            binary_image (ndarray): Binary image in BGR format
            boxes (list): List of boxes (x, y, w, h)
            contours (list): List of contours corresponding to boxes
            ear_position (str): "left" (Γ) or "right" (⅂)

        Returns:
            tuple: (filtered_boxes, filtered_contours, filtered_indices)
        """
        image_height, image_width = binary_image.shape[:2]
        filtered_boxes = []
        filtered_contours = []
        filtered_indices = []

        for i, (box, contour) in enumerate(zip(boxes, contours)):
            x, y, w, h = box
            print(f"\nAnalyzing box {i}: x={x}, y={y}, w={w}, h={h}")

            # Check edge proximity
            if y <= 5:
                print(f"Box {i} rejected: Too close to top edge (y={y})")
                continue
            if (image_height - (y + h) <= 5):
                print(f"Box {i} rejected: Too close to bottom edge (bottom margin={image_height - (y + h)})")
                continue
            if x <= 5:
                print(f"Box {i} rejected: Too close to left edge (x={x})")
                continue
            if (image_width - (x + w) <= 10):
                print(f"Box {i} rejected: Too close to right edge (right margin={image_width - (x + w)})")
                continue

            # Check size
            if (h <= 12) and (w <= 12):
                print(f"Box {i} rejected: Box too small (width={w}, height={h})")
                continue

            # Check aspect ratio
            if (h >= w * 3) and (w <= 15):
                print(f"Box {i} rejected: Extreme aspect ratio (height={h} >= 3*width={w})")
                continue

            # Extract box image
            box_image = self.extract_ear_from_binary_image(box, contour, i, ear_position, binary_image, show_image=True)
            #box_image = binary_image[y:y + h, x:x + w]
            if box_image.size == 0:
                print(f"Box {i} rejected: Empty box image")
                continue

            # Check white pixel distribution
            box_gray = box_image
            #box_gray = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
            total_pixels = box_gray.size
            white_pixels = np.sum(box_gray == 255)
            white_ratio = white_pixels / total_pixels

            #if white_ratio > 0.9:
            #    print(f"Box {i} rejected: Too many white pixels (ratio={white_ratio:.2f})")
            #    continue
            #if white_ratio < 0.1:
            #    print(f"Box {i} rejected: Too few white pixels (ratio={white_ratio:.2f})")
            #    continue

            # Split into quadrants and check distribution
            top_half = box_gray[0:h // 2, :]
            bottom_half = box_gray[h // 2:, :]
            top_left = box_gray[0:h // 2, 0:w // 2]
            top_right = box_gray[0:h // 2, w // 2:]
            bottom_left = box_gray[h // 2:, 0:w // 2]
            bottom_right = box_gray[h // 2:, w // 2:]

            # Calculate white pixel counts for each quadrant
            top_left_white = np.sum(top_left == 255)
            top_right_white = np.sum(top_right == 255)
            bottom_left_white = np.sum(bottom_left == 255)
            bottom_right_white = np.sum(bottom_right == 255)

            # Check distribution pattern
            #if ear_position == "left":  # Γ shape
            #    if  top_left_white > top_right_white:
            #        print(f"Box {i} rejected: Invalid left ear pattern")
            #        print(f"Top left whites: {top_left_white}, Top right whites: {top_right_white}")
                    #continue
            #else:  # ⅂ shape
            #    if top_left_white < top_right_white:
            #        print(f"Box {i} rejected: Invalid right ear pattern")
            #        print(f"Top left whites: {top_left_white}, Top right whites: {top_right_white}")
                    #continue

            # If passed all filters
            print(f"Box {i} accepted: Passed all filters")
            filtered_boxes.append(box)
            filtered_contours.append(contour)
            filtered_indices.append(i)

        print(f"\nFiltering summary:")
        print(f"Total boxes: {len(boxes)}")
        print(f"Filtered boxes: {len(filtered_boxes)}")
        print(f"Rejection rate: {(len(boxes) - len(filtered_boxes)) / len(boxes) * 100:.1f}%")

        return filtered_boxes, filtered_contours, filtered_indices

    def analyze_boxes(self, binary_image, boxes, contours, ear_position, show_results=False):
        """
        Analyzes boxes for ear patterns after filtering.
        """
        # First filter the boxes
        filtered_boxes, filtered_contours, filtered_indices = self.filter_boxes(
            binary_image, boxes, contours, ear_position
        )

        if not filtered_boxes:
            print(f"No valid {ear_position} ear candidates found after filtering")
            return -1, 0.0, binary_image

        # Continue with analysis on filtered boxes
        results = []
        annotated_image = binary_image.copy()
        best_score = -1
        best_box_index = -1

        for i, (box, contour) in enumerate(zip(filtered_boxes, filtered_contours)):
            score = self.analyze_single_contour(contour, ear_position)
            original_index = filtered_indices[i]
            results.append((original_index, score))

            if score > best_score:
                best_score = score
                best_box_index = original_index

            # Draw box and score
            x, y, w, h = box
            color = (0, 255, 0) if score >= self.min_score_threshold else (0, 0, 255)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated_image, f"{score:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if show_results:
                roi = binary_image[y:y + h, x:x + w]
                title = f"{ear_position} ear Box {original_index} (Score: {score:.2f})"
                cv2.imshow(title, roi)
                cv2.waitKey(0)

        return best_box_index, best_score, annotated_image

    def analyze_single_contour(self, contour, ear_position):
        """
        Analyzes a single contour for ear pattern characteristics.
        Returns a score between 0 and 1.
        """
        # Get bounding rect
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate scores for different characteristics
        shape_score = self._check_ear_shape(contour, ear_position)
        vertical_score = self._check_vertical_alignment(contour, ear_position)
        ratio_score = self._check_dimension_ratio(w, h)
        density_score = self._check_point_density(contour, ear_position)

        # Combine scores with weights
        final_score = (shape_score * 0.4 +
                       vertical_score * 0.3 +
                       ratio_score * 0.2 +
                       density_score * 0.1)

        return final_score

    def _check_ear_shape(self, contour, ear_position):
        """
        Checks if contour forms correct ear shape:
        Left ear: Γ shape (vertical on left, horizontal at bottom)
        Right ear: ⅂ shape (vertical on right, horizontal at bottom)
        """
        # Create a mask and fill the contour
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(mask, [shifted_contour], -1, (255, 255, 255), -1)

        # Split mask into regions
        top_half = mask[:h // 2, :]
        bottom_half = mask[h // 2:, :]

        # Calculate point distribution
        if ear_position == "left":  # Γ shape
            # Expect more points on right side at bottom
            bottom_left = np.sum(bottom_half[:, :w // 2]) / 255
            bottom_right = np.sum(bottom_half[:, w // 2:]) / 255
            horizontal_score = bottom_right / (bottom_left + 1) if bottom_left > 0 else 1.0

            # Expect points on left side at top
            top_left = np.sum(top_half[:, :w // 2]) / 255
            vertical_score = top_left / (np.sum(top_half) / 255) if np.sum(top_half) > 0 else 0.0

        else:  # ⅂ shape
            # Expect more points on left side at bottom
            bottom_left = np.sum(bottom_half[:, :w // 2]) / 255
            bottom_right = np.sum(bottom_half[:, w // 2:]) / 255
            horizontal_score = bottom_left / (bottom_right + 1) if bottom_right > 0 else 1.0

            # Expect points on right side at top
            top_right = np.sum(top_half[:, w // 2:]) / 255
            vertical_score = top_right / (np.sum(top_half) / 255) if np.sum(top_half) > 0 else 0.0

        return (horizontal_score * 0.6 + vertical_score * 0.4)

    def _check_vertical_alignment(self, contour, ear_position):
        """
        Checks if the vertical part is properly aligned.
        Left ear (Γ): check left side
        Right ear (⅂): check right side
        """
        x, y, w, h = cv2.boundingRect(contour)

        # Create mask and fill contour
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(mask, [shifted_contour], -1, (255, 255, 255), -1)

        # Check vertical distribution
        if ear_position == "left":  # Γ shape
            vertical_part = mask[:h // 2, :w // 3]  # Left third, top half
        else:  # ⅂ shape
            vertical_part = mask[:h // 2, -w // 3:]  # Right third, top half

        vertical_coverage = np.sum(vertical_part) / (vertical_part.size * 255)
        return min(1.0, vertical_coverage * 2)

    def _check_dimension_ratio(self, width, height):
        """
        Checks if width/height ratio is appropriate for an ear shape.
        Ideal ratio is around 1:1.5 to 1:2
        """
        if height == 0:
            return 0.0

        ratio = width / height

        # Score peaks at ratio 0.6 (1:1.67) and decreases towards 0.5 (1:2) and 1.0 (1:1)
        if ratio < 0.5 or ratio > 1.0:
            return 0.0
        elif ratio < 0.6:
            return 1.0 - abs(0.6 - ratio) * 2
        else:
            return 1.0 - abs(0.6 - ratio) * 2.5

    def _check_point_density(self, contour, ear_position):
        """
        Checks the density of points in key areas where we expect the ear pattern.
        """
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(mask, [shifted_contour], -1, (255, 255, 255), -1)

        # Define key areas
        if ear_position == "left":  # Γ shape
            key_area = mask[h // 2:, w // 2:]  # Bottom right quadrant
        else:  # ⅂ shape
            key_area = mask[h // 2:, :w // 2]  # Bottom left quadrant

        density = np.sum(key_area) / (key_area.size * 255)
        return min(1.0, density * 2)

class ImageProcessor(QThread):
    processing_complete = Signal(str, str)  # (result, status)
    image_display = Signal(np.ndarray)  # Signal to display images

    def __init__(self):
        super().__init__()
        self.temp_hist_extracted_jack_image = None
        self.hist_extracted_jack_image = None
        self.hist_extracted_jack_image_path = None
        self.image_path = None
        self.is_running = True

        # Class instances for processing
        self.ear_analyzer = EarPatternAnalyzer()

        # Directory paths for saving images
        self.raw_images_directory_path = None
        self.resized_images_directory_path = None
        self.binary_images_directory_path = None
        self.edge_images_directory_path = None
        self.ng_images_directory_path = None
        self.ok_images_directory_path = None
        self.adjusted_histogram_image = None

        # Config data
        self.config_dict = None

        # Detection results
        self.connector_lock_defect_check = True
        self.jack_fit_defect_check = True

        # Extracted part images
        self.extracted_jack_image_path = None
        self.extracted_jack_image = None
        self._extracted_jack_image = None
        self.jack_binary_image = None
        self.jack_binary_path = None
        self.pin_detected_image = None
        self.pin_detected_image_path = None

        # Other arguments
        self.longest_box = None

        # Flag for debugging
        self.is_images_shown = True

    def set_image_path(self, image_path):
        """Set the path of the image to process."""
        self.image_path = image_path

    def run(self):
        pass

    def preprocessing(self, image_path, show_image=False, apply_median_blur=True, median_kernel_size=5,
                      apply_gaussian_blur=False, gaussian_kernel_size=5, gaussian_sigma=0):
        """
        Preprocess the image: resize to 640x480 and apply noise reduction filters.

        Args:
            image_path (str): Path to the image to be processed
            show_image (bool): If True, display the processed image
            apply_median_blur (bool): If True, apply median blur filter
            median_kernel_size (int): Kernel size for median blur (must be odd)
            apply_gaussian_blur (bool): If True, apply Gaussian blur filter
            gaussian_kernel_size (int): Kernel size for Gaussian blur (must be odd)
            gaussian_sigma (float): Standard deviation for Gaussian blur

        Returns:
            Tuple[str, ndarray]: Path to the saved resized image and the processed image
        """
        print(f"Starting preprocessing for image: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image for preprocessing.")
            return None, None

        print("Image loaded successfully, processing...")

        # Resize the image
        resized_image = cv2.resize(image, (640, 480))
        processed_image = resized_image.copy()

        # Apply Median Blur if requested
        if apply_median_blur:
            # Ensure kernel size is odd
            median_kernel_size = median_kernel_size if median_kernel_size % 2 == 1 else median_kernel_size + 1
            processed_image = cv2.medianBlur(processed_image, median_kernel_size)
            if show_image:
                cv2.imshow("After Median Blur", processed_image)
                cv2.waitKey(1)

        # Apply Gaussian Blur if requested
        if apply_gaussian_blur:
            # Ensure kernel size is odd
            gaussian_kernel_size = gaussian_kernel_size if gaussian_kernel_size % 2 == 1 else gaussian_kernel_size + 1
            gaussian_kernel = (gaussian_kernel_size, gaussian_kernel_size)
            processed_image = cv2.GaussianBlur(processed_image, gaussian_kernel, gaussian_sigma)
            if show_image:
                cv2.imshow("After Gaussian Blur", processed_image)
                cv2.waitKey(1)

        # Create directory if it doesn't exist
        if not os.path.exists(self.resized_images_directory_path):
            print(f"Resized images directory does not exist. Creating: {self.resized_images_directory_path}")
            os.makedirs(self.resized_images_directory_path)

        # Create filename for the processed image
        base_name = os.path.basename(image_path)
        resized_image_path = os.path.join(self.resized_images_directory_path, f"processed_{base_name}")

        # Save the processed image
        success = cv2.imwrite(resized_image_path, processed_image)
        if success:
            print(f"Processed image saved at: {resized_image_path}")
        else:
            print("Failed to save the processed image.")
            return None, None

        # Show the processing steps if requested
        if show_image:
            # Create figure with subplots
            plt.figure(figsize=(15, 5))

            # Plot original resized image
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.title('Resized Image')
            plt.axis('off')

            # Plot processed image
            plt.subplot(132)
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title('Processed Image')
            plt.axis('off')

            # Plot difference
            difference = cv2.absdiff(resized_image, processed_image)
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
            plt.title('Difference')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        return resized_image_path, processed_image

    def convert_to_binary(self, gray_image, part_name, thresholds=[100, 150, 200], show_images=False,
                          apply_erosion=False, apply_dilation=False, apply_closing=False, dilation_iteration=1):
        """
        Convert grayscale image to binary using multiple thresholds, optionally apply erosion and dilation, and save the result.

        Args:
            gray_image (ndarray): Input grayscale image.
            part_name (str): Name of the part (e.g., "connector" or "jack") for naming the output file.
            thresholds (list): List of thresholds to apply.
            show_images (bool): If True, display each binary and processed image.
            apply_erosion (bool): If True, apply erosion to the binary image.
            apply_dilation (bool): If True, apply dilation to the binary image after erosion.

        Returns:
            Tuple[str, ndarray, Optional[ndarray]]: Path to the saved binary image, the binary image, and the processed image (if erosion/dilation is applied).
        """
        binary_images = []
        for threshold in thresholds:
            _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            binary_images.append(binary)
            self.image_display.emit(binary)  # Emit for display each binary image

        # Combine binary images
        combined_binary = np.zeros_like(binary_images[0])
        for binary in binary_images:
            combined_binary = cv2.bitwise_or(combined_binary, binary)

        # Apply erosion if requested
        processed_image = combined_binary
        if apply_erosion:
            kernel = np.ones((3, 3), np.uint8)  # Define the kernel for erosion
            processed_image = cv2.erode(processed_image, kernel, iterations=1)
            if show_images:
                cv2.imshow(f"Eroded Binary Image for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Apply dilation if requested
        if apply_dilation:
            # kernel = np.ones((3, 3), np.uint8)  # Define the kernel for dilation
            kernel = np.ones((3, 3), np.uint8)  # Define the kernel for dilation
            processed_image = cv2.dilate(processed_image, kernel, iterations=dilation_iteration)
            if show_images:
                cv2.imshow(f"Dilated Binary Image for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Apply closing if requested
        if apply_closing:
            kernel = np.ones((5, 5), np.uint8)  # Define the kernel for closing
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
            if show_images:
                cv2.imshow(f"Closing Morphology for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Construct the path to save the combined binary image
        if not os.path.exists(self.binary_images_directory_path):
            print(f"Binary images directory does not exist. Creating: {self.binary_images_directory_path}")
            os.makedirs(self.binary_images_directory_path)

        # Create a timestamped filename for the combined binary image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        binary_image_path = os.path.join(self.binary_images_directory_path, f"{part_name}_binary_image_{timestamp}.png")

        # Save the combined binary image
        success = cv2.imwrite(binary_image_path, combined_binary)
        if success:
            print(f"Binary image saved at: {binary_image_path}")
        else:
            print("Failed to save the binary image.")
            return None, combined_binary, processed_image  # Return None for path if saving failed

        # Show the combined binary image if requested
        if show_images:
            cv2.imshow(f"Combined Binary Image for {part_name}", combined_binary)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return binary_image_path, combined_binary, processed_image  # Return path, binary, and processed images

    def convert_to_binary_single_threshold(self, gray_image, part_name, threshold=150, show_images=False,
                                           apply_erosion=False, apply_dilation=False,
                                           apply_closing=False, apply_opening=False,
                                           erosion_iteration=1, dilation_iteration=1,
                                           is_binary_invert=False,
                                           use_adaptive_threshold=False,
                                           adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           adaptive_block_size=11, adaptive_c=2):
        """
        Convert a grayscale image to a binary image using either regular or adaptive thresholding.

        Args:
            gray_image (ndarray): Input grayscale image
            part_name (str): Name of the part for naming the output file
            threshold (int): Threshold value for regular thresholding
            show_images (bool): If True, display the binary and processed images
            apply_erosion (bool): If True, apply erosion to the binary image
            apply_dilation (bool): If True, apply dilation to the binary image
            apply_closing (bool): If True, apply closing morphology
            apply_opening (bool): If True, apply opening morphology
            erosion_iteration (int): Number of iterations for erosion
            dilation_iteration (int): Number of iterations for dilation
            is_binary_invert (bool): If True, invert the binary image
            use_adaptive_threshold (bool): If True, use adaptive thresholding instead of regular
            adaptive_method (int): Adaptive thresholding method
            adaptive_block_size (int): Block size for adaptive thresholding (must be odd)
            adaptive_c (int): Constant subtracted from mean or weighted mean

        Returns:
            Tuple[str, ndarray, ndarray]: Path to saved binary image, binary image, processed image
        """
        # Ensure grayscale input
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        if use_adaptive_threshold:
            # Ensure block size is odd
            adaptive_block_size = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1

            if is_binary_invert:
                binary_image = cv2.adaptiveThreshold(
                    gray_image,
                    255,
                    adaptive_method,
                    cv2.THRESH_BINARY_INV,
                    adaptive_block_size,
                    adaptive_c
                )
            else:
                binary_image = cv2.adaptiveThreshold(
                    gray_image,
                    255,
                    adaptive_method,
                    cv2.THRESH_BINARY,
                    adaptive_block_size,
                    adaptive_c
                )
        else:
            # Regular thresholding
            if is_binary_invert:
                _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
            else:
                _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        self.image_display.emit(binary_image)  # Emit for display the binary image
        processed_image = binary_image

        # Apply dilation if requested
        if apply_dilation:
            kernel = np.ones((3, 3), np.uint8)
            processed_image = cv2.dilate(processed_image, kernel, iterations=dilation_iteration)
            if show_images:
                cv2.imshow(f"Dilated Binary Image for {part_name}", processed_image)
                cv2.waitKey(0)

        # Apply erosion if requested
        if apply_erosion:
            kernel = np.ones((3, 3), np.uint8)
            processed_image = cv2.erode(processed_image, kernel, iterations=erosion_iteration)
            if show_images:
                cv2.imshow(f"Eroded Binary Image for {part_name}", processed_image)
                cv2.waitKey(0)

        # Apply opening if requested (erosion followed by dilation)
        if apply_opening:
            kernel = np.ones((3, 3), np.uint8)
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
            if show_images:
                cv2.imshow(f"Opening Morphology for {part_name}", processed_image)
                cv2.waitKey(0)

        # Apply closing if requested (dilation followed by erosion)
        if apply_closing:
            kernel = np.ones((3, 3), np.uint8)
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
            if show_images:
                cv2.imshow(f"Closing Morphology for {part_name}", processed_image)
                cv2.waitKey(0)

        # Create directory if it doesn't exist
        if not os.path.exists(self.binary_images_directory_path):
            print(f"Binary images directory does not exist. Creating: {self.binary_images_directory_path}")
            os.makedirs(self.binary_images_directory_path)

        # Create timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        binary_image_path = os.path.join(
            self.binary_images_directory_path,
            f"{part_name}_binary_image_{timestamp}.png"
        )

        # Save the processed image
        success = cv2.imwrite(binary_image_path, processed_image)
        if success:
            print(f"Binary image saved at: {binary_image_path}")
        else:
            print("Failed to save the binary image.")
            return None, binary_image, processed_image

        # Show the binary image if requested
        if show_images:
            cv2.imshow(f"Binary Image for {part_name}", binary_image)
            cv2.waitKey(0)

        return binary_image_path, binary_image, processed_image

    def extract_parts(self, gray_image, show_images=False, is_connector_saved=True):
        """
        Extract connector and FPC lead areas from the grayscale image based on configuration coordinates.

        Args:
            gray_image (ndarray): The grayscale image to process.
            show_images (bool): If True, display the extracted images.

        Returns:
            Tuple[str, str]: Paths of the saved images for connector and FPC lead.
        """
        # Retrieve coordinates from config_dict
        connector_coords = (
            self.config_dict["component-1-roi-coordinates"]["top-left-x"],
            self.config_dict["component-1-roi-coordinates"]["top-left-y"],
            self.config_dict["component-1-roi-coordinates"]["bottom-right-x"] -
            self.config_dict["component-1-roi-coordinates"]["top-left-x"],
            self.config_dict["component-1-roi-coordinates"]["bottom-right-y"] -
            self.config_dict["component-1-roi-coordinates"]["top-left-y"]
        )

        fpc_coords = (
            self.config_dict["component-2-roi-coordinates"]["top-left-x"],
            self.config_dict["component-2-roi-coordinates"]["top-left-y"],
            self.config_dict["component-2-roi-coordinates"]["bottom-right-x"] -
            self.config_dict["component-2-roi-coordinates"]["top-left-x"],
            self.config_dict["component-2-roi-coordinates"]["bottom-right-y"] -
            self.config_dict["component-2-roi-coordinates"]["top-left-y"]
        )

        # Extract images using the coordinates
        connector_image = gray_image[connector_coords[1]:connector_coords[1] + connector_coords[3],
                          connector_coords[0]:connector_coords[0] + connector_coords[2]]
        fpc_image = gray_image[fpc_coords[1]:fpc_coords[1] + fpc_coords[3],
                    fpc_coords[0]:fpc_coords[0] + fpc_coords[2]]

        # Create a timestamp for the filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create paths for saving the images with timestamp
        connector_image_path = os.path.join(self.resized_images_directory_path, f"connector_image_{timestamp}.png")
        fpc_image_path = os.path.join(self.resized_images_directory_path, f"fpc_image_{timestamp}.png")

        # Save the extracted images
        if is_connector_saved:
            cv2.imwrite(connector_image_path, connector_image)
            print(f"Connector image saved at: {connector_image_path}")

        cv2.imwrite(fpc_image_path, fpc_image)
        print(f"FPC image saved at: {fpc_image_path}")

        # Show the extracted images if requested
        if show_images:
            cv2.imshow("Connector Image", connector_image)
            cv2.imshow("FPC Image", fpc_image)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return connector_image_path, fpc_image_path, connector_image, fpc_image  # Return the paths of the saved images

    def add_black_border(self, image, border_size=5):
        """
        Adds a black border around the image.

        Args:
            image (ndarray): The original image.
            border_size (int): The size of the border to add.

        Returns:
            ndarray: The image with a black border added.
        """
        return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                  value=(0, 0, 0))

    def add_white_border(self, image, border_size=5):
        """
        Adds a white border around the image.

        Args:
            image (ndarray): The original image.
            border_size (int): The size of the border to add.

        Returns:
            ndarray: The image with a black border added.
        """
        return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))

    def find_contours(self, binary_image, show_image=False):
        """
        Find contours in the binary image and return the contours and information.

        Args:
            binary_image (ndarray): The binary image to process.
            show_image (bool): If True, display the input image with contours drawn.

        Returns:
            List[ndarray]: Contours found in the binary image.
            List[Dict]: Information about each contour (length, area).
        """
        if len(binary_image.shape) == 3:
            gray_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        elif len(binary_image.shape) != 2:
            raise ValueError("Input must be a binary or single-channel grayscale image.")

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_info = []

        for contour in contours:
            length = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            contour_info.append({'length': length, 'area': area})

        if show_image:
            contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            cv2.imshow("Contours", contour_image)
            cv2.waitKey(0)

        return contours, contour_info

    def find_and_draw_longest_contour(self, contours, image, color=(255, 0, 0), thickness=2):
        """
        Finds the bounding box with the largest width and height from the given contours
        and draws it on the image.

        Args:
            contours (List[ndarray]): List of contours to analyze.
            image (ndarray): The image on which to draw the bounding box.
            color (Tuple[int, int, int]): The color of the bounding box in BGR format.
            thickness (int): The thickness of the bounding box lines.

        Returns:
            Tuple[float, Tuple[int, int, int, int], ndarray]:
                The area of the largest bounding box, its bounding box (x, y, w, h), and the modified image.
        """
        largest_box = (0, 0, 0, 0)  # (x, y, w, h)
        max_area = 0  # Initialize maximum area

        for contour in contours:
            if contour.size > 0:  # Ensure the contour is not empty
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h  # Calculate area of the bounding box

                # Check if this bounding box has a larger area than the current max
                if area > max_area:
                    max_area = area
                    largest_box = (x, y, w, h)

        if largest_box != (0, 0, 0, 0):
            # Draw the bounding box on the image
            x, y, w, h = largest_box
            print("Longest contour box: width: " + str(w) + ", height: " + str(h) + ", x: " + str(x) + " ,y: " + str(y))

            # Re-assign values to the box if it's not the desired box
            if h < 113:
                print("Missed recognize longest box. Re calculate and draw")
                h += 34
                y -= 34
                max_area = w * h
                largest_box = (x, y, w, h)

            if w < 386:
                print("Missed recognize longest box. Re calculate and draw")
                w += 25
                x -= 25
                max_area = w * h
                largest_box = (x, y, w, h)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            return max_area, largest_box, image
        else:
            print("No contours found.")
            return 0, (0, 0, 0, 0), image  # Return zeroed box coordinates when no contour is found

    def filter_and_draw_bounding_boxes(self, contours, longest_box, image, color=(0, 255, 0), thickness=2):
        """
        Filters bounding boxes that are inside the longest bounding box and draws them on the image.

        Args:
            contours (List[ndarray]): List of contours to analyze.
            longest_box (Tuple[int, int, int, int]): The bounding box (x, y, w, h) of the longest contour.
            image (ndarray): The image on which to draw the bounding boxes.
            color (Tuple[int, int, int]): The color of the bounding boxes in BGR format.
            thickness (int): The thickness of the bounding box lines.

        Returns:
            Tuple[ndarray, List[ndarray]]: The modified image with bounding boxes drawn and a list of filtered contours.
        """
        longest_x, longest_y, longest_w, longest_h = longest_box
        filtered_contours = []

        for contour in contours:
            box = cv2.boundingRect(contour)
            x, y, w, h = box

            # Check if the bounding box is completely inside the longest box
            if (x >= longest_x and y >= longest_y and
                    x + w <= longest_x + longest_w and
                    y + h <= longest_y + longest_h):
                filtered_contours.append(contour)  # Store the original contour

        # Draw the filtered bounding boxes on the image
        for contour in filtered_contours:
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        return image, filtered_contours

    def draw_bounding_boxes(self, image, contours, show_image=False):
        """
        Draw bounding boxes around contours in the image.

        Args:
            image (ndarray): The image on which to draw bounding boxes.
            contours (List[ndarray]): List of contours.
            show_image (bool): If True, display the image with bounding boxes drawn.

        Returns:
            ndarray: The image with bounding boxes drawn.
        """
        boxed_image = image.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print("Drawing bounding box with height: " + str(h))
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if show_image:
            cv2.imshow("Drawn Bounding Boxes", boxed_image)
            cv2.waitKey(0)

        return boxed_image

    def draw_bounding_box(self, image, box, color=(0, 255, 0), thickness=2):
        """
        Draws bounding boxes on the given image.

        Parameters:
        - image: Input image (as a NumPy array).
        - boxes: List of boxes, where each box is represented as [x, y, width, height].
        - color: Color of the bounding boxes (default is green).
        - thickness: Thickness of the bounding box lines (default is 2).

        Returns:
        - The image with bounding boxes drawn on it.
        """
        x, y, width, height = box
        # Draw the rectangle on the image
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

        return image

    def draw_min_area_rect(self, binary_image, show_image=False):
        """
        Finds the largest contour in a binary/grayscale image and draws its minimum area rectangle
        along with regular bounding box.

        Args:
            binary_image (ndarray): Input binary/grayscale image or BGR image
            show_image (bool): If True, displays the result image with both boxes drawn

        Returns:
            Tuple[ndarray, float, Tuple[int, int, int, int], Tuple[float, float, float, float]]:
                - The image with boxes drawn
                - The tilt angle of the min area rectangle
                - Regular bounding box (x, y, w, h)
                - Min area rectangle (center_x, center_y, width, height)
        """
        # Check if input is BGR or grayscale
        if len(binary_image.shape) == 3:
            # Input is BGR
            display_image = binary_image.copy()
            # Convert to grayscale for contour detection
            binary_for_contours = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        else:
            # Input is grayscale/binary
            binary_for_contours = binary_image
            display_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        # Find all contours directly from binary image
        contours, _ = self.find_contours(binary_for_contours, show_image=False)

        if not contours:
            print("No contours found in the image")
            return display_image, 0, (0, 0, 0, 0), (0, 0, 0, 0)

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get regular bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw regular bounding box in green
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get rotated rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Changed from np.int0 to np.int32

        # Draw rotated rectangle in blue
        cv2.drawContours(display_image, [box], 0, (255, 0, 0), 2)

        # Calculate and draw angle
        center = rect[0]
        angle = rect[2]

        # Normalize angle to be between -90 and 90 degrees
        if angle < -45:
            angle = 90 + angle
        if angle > 45:
            angle = angle - 90

        # Draw angle text
        text = f"Angle: {angle:.1f} degrees"
        cv2.putText(display_image, text,
                    (int(center[0]) - 50, int(center[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if show_image:
            cv2.imshow("Min Area Rect", display_image)
            cv2.waitKey(0)

        # Return the image with boxes drawn, angle, and box coordinates
        return (display_image,
                angle,
                (x, y, w, h),  # Regular bounding box
                (center[0], center[1], rect[1][0], rect[1][1]))  # Min area rect (center_x, center_y, width, height)

    def draw_ear_bounding_boxes(self, image, boxes, name, color=(0, 255, 0), thickness=2):
        """
        Draws bounding boxes on the given image.

        Parameters:
        - image: Input image (as a NumPy array).
        - boxes: List of boxes, where each box is represented as [x, y, width, height].
        - color: Color of the bounding boxes (default is green).
        - thickness: Thickness of the bounding box lines (default is 2).

        Returns:
        - The image with bounding boxes drawn on it.
        """
        for box in boxes:
            x, y, width, height = box
            # Draw the rectangle on the image
            cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

        if self.is_images_shown:
            cv2.imshow("Ear Boxes:" + name, image)
            cv2.waitKey(0)

        return image

    def analyze_histogram(self, image_path, show_plot=False):
        """
        Analyzes and displays the histogram of an image from the given path.

        Args:
            image_path (str): Path to the input image
            show_plot (bool): If True, displays the histogram plot

        Returns:
            dict: Dictionary containing histogram analysis results:
                - 'mean': Mean pixel intensity
                - 'std': Standard deviation of pixel intensities
                - 'median': Median pixel intensity
                - 'min': Minimum pixel intensity
                - 'max': Maximum pixel intensity
                - 'histogram': Array of histogram values
                - 'bins': Array of bin edges
                - 'brightness_judgment': String indicating if image is 'too_dark', 'too_bright', or 'normal'
        """
        # Load and convert image to grayscale
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None

        try:
            # Read image and convert to grayscale if needed
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Failed to load image at {image_path}")
                return None

            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten()  # Convert to 1D array

        # Calculate basic statistics
        mean_intensity = np.mean(gray_image)
        std_intensity = np.std(gray_image)
        median_intensity = np.median(gray_image)
        min_intensity = np.min(gray_image)
        max_intensity = np.max(gray_image)

        # Define thresholds for brightness judgment
        DARK_THRESHOLD = 50
        BRIGHT_THRESHOLD = 200

        # Make brightness judgment
        if mean_intensity < DARK_THRESHOLD:
            brightness_judgment = 'too_dark'
        elif mean_intensity > BRIGHT_THRESHOLD:
            brightness_judgment = 'too_bright'
        else:
            brightness_judgment = 'normal'

        # Find peaks in histogram (local maxima)
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((i, hist[i]))

        # Sort peaks by height (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)

        # If show_plot is True, create and display the histogram
        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(hist, color='blue')
            plt.title(f'Grayscale Histogram: {os.path.basename(image_path)}\nMean Intensity: {mean_intensity:.1f}')
            plt.xlabel('Pixel Intensity (0=Black, 255=White)')
            plt.ylabel('Frequency (Pixel Count)')

            # Add vertical lines for thresholds and statistics
            plt.axvline(DARK_THRESHOLD, color='red', linestyle='--', label='Dark Threshold')
            plt.axvline(BRIGHT_THRESHOLD, color='red', linestyle='--', label='Bright Threshold')
            plt.axvline(mean_intensity, color='green', linestyle='-', label='Mean Intensity')
            plt.axvline(median_intensity, color='yellow', linestyle='--', label='Median')

            # Add text box with statistics and judgment
            stats_text = f'Statistics:\n' \
                         f'Mean: {mean_intensity:.1f}\n' \
                         f'Std Dev: {std_intensity:.1f}\n' \
                         f'Median: {median_intensity:.1f}\n' \
                         f'Min: {min_intensity}\n' \
                         f'Max: {max_intensity}\n' \
                         f'Brightness: {brightness_judgment}'

            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Mark the top 3 peaks
            for i, (intensity, count) in enumerate(peaks[:3]):
                plt.plot(intensity, count, 'ro')
                plt.annotate(f'Peak {i + 1}: {intensity}',
                             xy=(intensity, count),
                             xytext=(10, 10),
                             textcoords='offset points')

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        # Prepare results dictionary
        results = {
            'mean': mean_intensity,
            'std': std_intensity,
            'median': median_intensity,
            'min': min_intensity,
            'max': max_intensity,
            'histogram': hist,
            'bins': np.arange(256),
            'peaks': peaks[:3],  # Top 3 peaks
            'brightness_judgment': brightness_judgment
        }

        # Print analysis results
        print(f"\nHistogram Analysis Results for {os.path.basename(image_path)}:")
        print(f"Mean Intensity: {mean_intensity:.1f}")
        print(f"Standard Deviation: {std_intensity:.1f}")
        print(f"Median Intensity: {median_intensity:.1f}")
        print(f"Intensity Range: [{min_intensity}, {max_intensity}]")
        print(f"Brightness Judgment: {brightness_judgment}")
        print("\nTop 3 Peaks (Intensity, Count):")
        for i, (intensity, count) in enumerate(peaks[:3]):
            print(f"Peak {i + 1}: Intensity = {intensity}, Count = {count:.0f}")

        return results

    def adjust_histogram(self, image_path, show_result=False):
        """
        Adjusts the histogram of an image to balance brightness when peaks are too close
        to dark or bright thresholds.

        Args:
            image_path (str): Path to the input image
            show_result (bool): If True, displays original and adjusted images side by side

        Returns:
            Tuple[ndarray, str]: Adjusted image and the path where it was saved
        """
        # Load image and check validity
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None, None

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Failed to load image at {image_path}")
                return None, None

            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
                image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None

        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Find peaks
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((i, hist[i]))

        # Sort peaks by height (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Define thresholds
        DARK_THRESHOLD = 50
        BRIGHT_THRESHOLD = 200

        # Get the main peaks (top 2)
        main_peaks = peaks[:2] if len(peaks) >= 2 else peaks
        peak_intensities = [p[0] for p in main_peaks]

        # Determine if adjustment is needed
        needs_adjustment = False
        is_dark = any(p < DARK_THRESHOLD + 30 for p in peak_intensities)
        is_bright = any(p > BRIGHT_THRESHOLD - 30 for p in peak_intensities)

        adjusted_image = image.copy()

        if is_dark or is_bright:
            needs_adjustment = True
            if is_dark:
                # Increase brightness and contrast
                adjusted_gray = cv2.convertScaleAbs(gray_image, alpha=1.3, beta=30)
            elif is_bright:
                # Decrease brightness and increase contrast
                adjusted_gray = cv2.convertScaleAbs(gray_image, alpha=1.2, beta=-30)

            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            adjusted_gray = clahe.apply(adjusted_gray)

            # Convert back to BGR if original was color
            if len(image.shape) == 3:
                adjusted_image = cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)
            else:
                adjusted_image = adjusted_gray

        # Save the adjusted image
        output_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_adjusted{ext}")
        self.adjusted_histogram_image = adjusted_gray
        cv2.imwrite(output_path, adjusted_image)

        if show_result:
            # Create figure with subplots
            plt.figure(figsize=(15, 5))

            # Plot original image
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            # Plot adjusted image
            plt.subplot(132)
            plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
            plt.title('Adjusted Image')
            plt.axis('off')

            # Plot histograms
            plt.subplot(133)
            plt.plot(cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten(),
                     color='blue', alpha=0.7, label='Original')
            plt.plot(cv2.calcHist([cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)],
                                  [0], None, [256], [0, 256]).flatten(),
                     color='red', alpha=0.7, label='Adjusted')

            plt.axvline(DARK_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(BRIGHT_THRESHOLD, color='gray', linestyle='--', alpha=0.5)

            plt.title('Histogram Comparison')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()

            # Show analysis of adjusted image
            self.analyze_histogram(output_path, show_plot=True)

        adjustment_type = "balanced (no adjustment needed)"
        if needs_adjustment:
            if is_dark:
                adjustment_type = "brightened and contrast enhanced"
            elif is_bright:
                adjustment_type = "darkened and contrast enhanced"

        print(f"\nImage Adjustment Results:")
        print(f"Input image: {os.path.basename(image_path)}")
        print(f"Adjustment type: {adjustment_type}")
        print(f"Output saved to: {output_path}")

        return adjusted_image, output_path

    def find_edges(self, input_image, show_image=False):
        """
        Convert the input image to an edge image using Canny edge detection.

        Args:
            input_image (ndarray): The input image to process.
            show_image (bool): If True, display the processed edge image.

        Returns:
            ndarray: The edge image.
        """
        # Check if the input image is color; if so, convert to grayscale
        if len(input_image.shape) == 3:
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = input_image  # Already grayscale

        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, 100, 200)  # Thresholds can be adjusted

        # Show the edge image if requested
        if show_image:
            cv2.imshow("Edge Image", edges)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return edges  # Return the edge image

    def find_edges_invert(self, input_image, show_image=False):
        """
        Convert the input image to an edge image using Canny edge detection,
        optimized for dark objects on light background.

        Args:
            input_image (ndarray): The input image to process
            show_image (bool): If True, display the processed edge image

        Returns:
            ndarray: The edge image
        """
        # Check if the input image is color; if so, convert to grayscale
        if len(input_image.shape) == 3:
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = input_image  # Already grayscale

        # Invert the image so dark objects become light
        inverted_image = cv2.bitwise_not(gray_image)

        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 1.5)

        # Apply Canny edge detection with adjusted thresholds
        edges = cv2.Canny(blurred_image, 100, 200)  # Thresholds can be adjusted

        # Show the processing steps if requested
        if show_image:
            # Create figure with subplots
            plt.figure(figsize=(15, 5))

            # Plot original image
            plt.subplot(131)
            plt.imshow(gray_image, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            # Plot inverted image
            plt.subplot(132)
            plt.imshow(inverted_image, cmap='gray')
            plt.title('Inverted Image')
            plt.axis('off')

            # Plot edge image
            plt.subplot(133)
            plt.imshow(edges, cmap='gray')
            plt.title('Edge Detection')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        return edges

    def non_max_suppression(self, groups, overlap_threshold=0.001):
        def compute_iou(box1, box2):
            """Compute Intersection over Union (IoU) between two boxes."""
            # Extract coordinates
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Convert to x1, y1, x2, y2 format
            box1_x2, box1_y2 = x1 + w1, y1 + h1
            box2_x2, box2_y2 = x2 + w2, y2 + h2

            # Calculate intersection coordinates
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(box1_x2, box2_x2)
            y_bottom = min(box1_y2, box2_y2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # Calculate intersection area
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # Calculate union area
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - intersection_area

            # Calculate IoU
            iou = intersection_area / union_area if union_area > 0 else 0.0

            return iou

        def suppress(boxes):
            """Apply non-max suppression to a single group of boxes."""
            if len(boxes) == 0:
                return []

            # Convert boxes to numpy array
            boxes = np.array(boxes)

            # Initialize list to keep track of selected boxes
            selected_indices = []

            # Sort boxes by area (largest first)
            areas = boxes[:, 2] * boxes[:, 3]  # width * height
            sorted_indices = np.argsort(-areas)  # Negative for descending order

            while len(sorted_indices) > 0:
                # Select the largest box
                current_idx = sorted_indices[0]

                # Add the current index to selected indices
                selected_indices.append(current_idx)

                # If this is the last box, break
                if len(sorted_indices) == 1:
                    break

                # Compare the current box with all remaining boxes
                ious = []
                for idx in sorted_indices[1:]:
                    iou = compute_iou(boxes[current_idx], boxes[idx])
                    ious.append(iou)

                # Convert to numpy array for boolean indexing
                ious = np.array(ious)

                # Keep only boxes with IoU less than threshold
                sorted_indices = sorted_indices[1:][ious < overlap_threshold]

            # Return selected boxes
            return boxes[selected_indices].tolist()

        def filter_nearby_boxes(boxes, distance_threshold=10):
            """Filter boxes that are very close to each other."""
            if len(boxes) == 0:
                return []

            filtered_boxes = []
            boxes = np.array(boxes)

            # Sort boxes by x-coordinate
            sorted_indices = np.argsort(boxes[:, 0])
            boxes = boxes[sorted_indices]

            current_box = boxes[0]
            filtered_boxes.append(current_box.tolist())

            for box in boxes[1:]:
                # Calculate center points
                current_center_x = current_box[0] + current_box[2] / 2
                current_center_y = current_box[1] + current_box[3] / 2
                box_center_x = box[0] + box[2] / 2
                box_center_y = box[1] + box[3] / 2

                # Calculate distance between centers
                distance = np.sqrt((current_center_x - box_center_x) ** 2 +
                                   (current_center_y - box_center_y) ** 2)

                if distance > distance_threshold:
                    filtered_boxes.append(box.tolist())
                    current_box = box

            return filtered_boxes

        # Process each group
        filtered_groups = []
        for group in groups:
            # First apply non-max suppression
            suppressed_boxes = suppress(group)
            # Then filter nearby boxes
            final_boxes = filter_nearby_boxes(suppressed_boxes)
            filtered_groups.append(final_boxes)

            # Debug prints
            print(f"Original boxes in group: {len(group)}")
            print(f"After NMS: {len(suppressed_boxes)}")
            print(f"After nearby filtering: {len(final_boxes)}")

        return filtered_groups

    def boxes_overlap(self, box1, box2):
        """Check if two boxes overlap."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def filter_overlapping_boxes(self, groups):
        """Filter out overlapping boxes from the groups."""
        cleaned_groups = []

        for group in groups:
            non_overlapping_boxes = []
            for box in group:
                overlap_found = False

                for other_box in group:
                    if box is not other_box:  # Skip self-comparison
                        if self.boxes_overlap(box, other_box):
                            overlap_found = True
                            break

                if not overlap_found:
                    non_overlapping_boxes.append(box)

            cleaned_groups.append(non_overlapping_boxes)

        return cleaned_groups

    def calculate_x_distances(self, boxes):
        """Calculate the minimum and maximum x distances between boxes."""
        if len(boxes) < 2:
            return None, None  # Not enough boxes to calculate distances

        x_values = [box[0] for box in boxes]  # Extract x values
        min_x = min(x_values)
        max_x = max(x_values)

        # Calculate distances
        distances = []
        for i in range(len(x_values)):
            for j in range(i + 1, len(x_values)):
                distances.append(abs(x_values[i] - x_values[j]))

        if distances:
            min_distance = min(distances)
            max_distance = max(distances)
        else:
            min_distance = max_distance = None

        return min_distance, max_distance

    def filter_far_away_boxes(self, groups, x_threshold):
        """Filter out boxes with x values that are too far away."""
        filtered_groups = []

        for group in groups:
            min_distance, max_distance = self.calculate_x_distances(group)

            if min_distance is None or max_distance is None:
                filtered_groups.append(group)  # Return the group as is if no distances
                continue

            # Check if the maximum x value box is more than twice the minimum distance
            max_box = max(group, key=lambda box: box[0])  # Box with max x value
            if max_box[0] > 2 * min_distance:
                group.remove(max_box)  # Remove the max x value box

            filtered_groups.append(group)

        return filtered_groups

    def clean_groups(self, _groups):
        cleaned_groups = []

        for group in _groups:
            cleaned_group = []

            for box in group:
                x, y, w, h = box  # Unpack the box values
                is_inside = False

                # Check if the current box is inside any other box in the group
                for other_box in group:
                    if box is not other_box:  # Skip the same box
                        other_x, other_y, other_w, other_h = other_box

                        # Check if the current box is inside the other box
                        if (x >= other_x and
                                y >= other_y and
                                x + w <= other_x + other_w and
                                y + h <= other_y + other_h):
                            is_inside = True
                            break  # No need to check further

                if not is_inside:
                    cleaned_group.append(box)  # Keep the box if it's not inside another box

            cleaned_groups.append(cleaned_group)  # Add cleaned group to the result

        return cleaned_groups

    def count_pixel_distribution_in_box(self, binary_image, box):
        """
        Counts the distribution of black and white pixels within a specified bounding box in a binary image.

        Parameters:
        - binary_image: Input binary image (as a NumPy array).
        - box: A list or tuple containing the box parameters [x, y, width, height].

        Returns:
        - black_pixel_percentage: Percentage of black pixels (value 0) within the box.
        - white_pixel_percentage: Percentage of white pixels (value 255) within the box.
        """
        x, y, w, h = box

        # Crop the image to the specified box
        cropped_image = binary_image[y:y + h, x:x + w]

        # Count total pixels in the cropped area
        total_pixels = cropped_image.size

        # Count black and white pixels in the cropped image
        black_pixel_count = np.sum(cropped_image == 0)
        white_pixel_count = np.sum(cropped_image == 255)

        # Calculate percentages
        black_pixel_percentage = (black_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        white_pixel_percentage = (white_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

        return black_pixel_percentage, white_pixel_percentage

    def old_check_ear_pattern(self, binary_image, box, ear_position, white_pixel_threshold=0.1, show_parts=False):
        """
        Checks for the presence of a specific ear pattern in the specified bounding box of a binary image.

        Parameters:
        - binary_image: Input binary image (as a NumPy array).
        - box: A tuple or list containing the box parameters (x, y, width, height).
        - ear_position: The position of the ear ("left" or "right").
        - white_pixel_threshold: Minimum white pixel distribution (percentage) required for pattern validation.
        - show_parts: Boolean indicating whether to display the extracted parts of the image.

        Returns:
        - bool: True if the pattern matches the criteria, False otherwise.
        """
        # Unpack the box parameters
        x, y, width, height = box

        # Extract the specified box from the binary image
        box_image = binary_image[y:y + height, x:x + width]

        # Ensure the box image is valid
        if box_image.size == 0:
            raise ValueError("Extracted box is empty. Check the input box coordinates.")

        # Get the dimensions of the box image
        box_height = height
        box_width = width

        # Define the parts of the box image
        top_half = box_image[0:box_height // 2, :]
        bottom_half = box_image[box_height // 2:, :]
        left_half = box_image[:, 0:box_width // 2]
        right_half = box_image[:, box_width // 2:]

        # Determine which top part to check based on ear position
        if ear_position == "left":
            top_part_to_check = top_half[:, 0:box_width // 2]
            other_top_part = top_half[:, box_width // 2:]
        elif ear_position == "right":
            top_part_to_check = top_half[:, box_width // 2:]
            other_top_part = top_half[:, 0:box_width // 2]
        else:
            raise ValueError("Invalid ear_position. Must be 'left' or 'right'.")

        # Calculate white pixel distribution in the specified parts
        white_pixel_ratio_top_to_check = np.sum(top_part_to_check == 255) / top_part_to_check.size
        white_pixel_ratio_other_top = np.sum(other_top_part == 255) / other_top_part.size

        # Show parts if requested
        if show_parts:
            cv2.imshow("Top Part to Check", top_part_to_check)
            cv2.imshow("Other Top Part", other_top_part)
            cv2.imshow("Bottom Half", bottom_half)
            cv2.imshow("Left Half", left_half)
            cv2.imshow("Right Half", right_half)
            cv2.waitKey(0)  # Wait for a key press to close the windows
            cv2.destroyAllWindows()

        # Check if top part matches the white pixel threshold and other part doesn't
        if white_pixel_ratio_top_to_check >= white_pixel_threshold and white_pixel_ratio_other_top < white_pixel_threshold:
            return True

        # Check if all four parts have white pixels
        all_parts_have_white = (
                np.any(top_half == 255) and
                np.any(bottom_half == 255) and
                np.any(left_half == 255) and
                np.any(right_half == 255)
        )

        # Check if the box is almost entirely filled with white
        total_white_ratio = np.sum(box_image == 255) / box_image.size
        if total_white_ratio > 0.9:
            # Determine if the box forms a line-like structure
            row_white_ratios = np.sum(box_image == 255, axis=1) / box_width
            col_white_ratios = np.sum(box_image == 255, axis=0) / box_height

            is_horizontal_line = np.mean(row_white_ratios > 0.9) > 0.8  # Most rows are almost fully white
            is_vertical_line = np.mean(col_white_ratios > 0.9) > 0.8  # Most columns are almost fully white

            # If it is a line, return False
            if is_horizontal_line or is_vertical_line:
                return False

        # Return True if all parts have white pixels and the box is not a line
        return all_parts_have_white

    def extract_ear_from_binary_image(self, box, contours, box_index, ear_position, binary_image, show_image=False):
        """
        Extracts an ear object from a binary image based on the given box and contour information.

        Args:
            box (tuple): Box coordinates (x, y, w, h)
            contours (list): List of contours detected in the binary image
            box_index (int): Index of the contour/box to process
            ear_position (str): Position identifier ("left" or "right")
            binary_image (ndarray): Input binary image (grayscale or BGR)
            show_image (bool): If True, displays the extracted object

        Returns:
            ndarray: Extracted ear object as a binary image
        """
        # Check if image is BGR and convert to grayscale if needed
        if len(binary_image.shape) == 3:
            print("Converting BGR image to grayscale")
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        elif len(binary_image.shape) != 2:
            print("Error: Input image must be either grayscale or BGR")
            return None

        # Create a mask of zeros with same size as input image
        mask = np.zeros(binary_image.shape, dtype=np.uint8)

        # Get the contour corresponding to the box_index
        if 0 <= box_index < len(contours):
            target_contour = contours[box_index]

            # Draw the contour on the mask in white
            cv2.drawContours(mask, [target_contour], -1, (255, 255, 255), -1)

            # Extract box coordinates
            x, y, w, h = box

            # Create a region of interest (ROI) using the box coordinates
            roi_mask = mask[y:y + h, x:x + w]
            roi_binary = binary_image[y:y + h, x:x + w]

            # Apply the mask to get only the object
            extracted_object = cv2.bitwise_and(roi_binary, roi_binary, mask=roi_mask)

            if show_image:
                window_title = f"Extracted {ear_position} ear object (Box {box_index})"
                cv2.imshow(window_title, roi_binary)
                cv2.waitKey(0)

            return roi_binary
        else:
            print(f"Error: Box index {box_index} is out of range for contours list")
            return None

    def check_ear_pattern(self, binary_image, box, box_contour, box_index, ear_position, show_parts=True) -> bool:
        """
        Checks for the presence of a specific ear pattern in the specified bounding box of a binary image.

        Parameters:
        - binary_image: Input binary image (as a NumPy array).
        - box: A tuple or list containing the box parameters (x, y, width, height).
        - ear_position: The position of the ear ("left" or "right").
        - show_parts: Boolean indicating whether to display the extracted parts of the image.

        Returns:
        - bool: True if the top part to check has a higher white pixel distribution than the other top part,
                or if all parts have white pixels and the box is not a line, False otherwise.
        """
        # Unpack the box parameters
        x, y, width, height = box
        #print("Ear box width: " + str(width) + " , height: " + str(height))

        # Extract the specified box from the binary image
        #box_image = binary_image[y:y + height, x:x + width]
        extracted_ear_image = self.extract_ear_from_binary_image(box, box_contour, box_index, ear_position,
                                                                    binary_image, show_image=True)

        image_height, image_width = binary_image.shape[:2]

        # Check if box image includes more than 1 contour
        # box_image_edges = self.find_edges(box_image, show_image=False)
        # box_image_contour, box_image_contour_info = self.find_contours(box_image_edges, show_image=False)

        # if len(box_image_contour) > 1:
        #    print("It seems there are more than 1 ear pattern in the box. Return False")
        # return False

        # Return False if box size is almost fit to image size
        if (width / image_width >= 0.90) or (height / image_height >= 0.90):
            print("Box is almost fit to image size. Return False")
            return False

        # Return False if box y is very close to 0
        if y <= 5 or (image_height - (y + height) <= 10):
            print("Box Y or height value is very close to top edge. Return False")
            return False
        else:
            print("Box Y value is ok. Y: " + str(y) + " - Height: " + str(image_height - (y + height)))

        # Return False if box x is very close to 0 or width
        if x <= 5 or (image_width - (x + width) <= 10):
            print("Box X value or width is very close to 0 or the edge. Return False")
            return False
        else:
            print("Box X value is ok. X: " + str(x) + " - Width: " + str(image_width - (x + width)))

        # Ensure the box image is valid
        if extracted_ear_image.size == 0:
            raise ValueError("Extracted box is empty. Check the input box coordinates.")

        # Get the dimensions of the box image
        box_height = height
        box_width = width

        print("Ear box width: " + str(box_width))
        print("Ear box height: " + str(box_height))

        if (box_height <= 8) and (box_width <= 8):
            print("Box is too small to be potential ear. Return False")
            return False

        if (box_height >= box_width * 4) and (box_width <= 12):
            print("Box height and width are too different. Return False")
            return False

        # Define the four parts of the box image
        top_half = extracted_ear_image[0:box_height // 2, :]  # Full width, half height
        bottom_half = extracted_ear_image[box_height // 2:, :]
        top_left_half = extracted_ear_image[0:box_height // 2, 0:box_width // 2]
        top_right_half = extracted_ear_image[0:box_height // 2, box_width // 2:]
        bottom_left_half = extracted_ear_image[box_height // 2:, 0:box_width // 2]
        bottom_right_half = extracted_ear_image[box_height // 2:, box_width // 2:]

        # Show parts if requested
        if show_parts:
            cv2.imshow("Top Half " + ear_position + " - " + str(box_index), top_half)
            cv2.imshow("Top Left Half " + ear_position + " - " + str(box_index), top_left_half)
            cv2.imshow("Top Right Half " + ear_position + " - " + str(box_index), top_right_half)
            cv2.imshow("Bottom Left Half " + ear_position + " - " + str(box_index), bottom_left_half)
            cv2.imshow("Bottom Right Half " + ear_position + " - " + str(box_index), bottom_right_half)
            cv2.waitKey(0)  # Wait for a key press to close the windows

            # Check if all four parts have white pixels
            all_parts_have_white = (
                    np.any(top_left_half == 255) and
                    np.any(bottom_left_half == 255) and
                    np.any(top_right_half == 255) and
                    np.any(bottom_right_half == 255)
            )

            # Check if the box is almost entirely filled with white
            total_white_ratio = np.sum(extracted_ear_image == 255) / extracted_ear_image.size
            if total_white_ratio > 0.9:
                # Determine if the box forms a line-like structure
                row_white_ratios = np.sum(extracted_ear_image == 255, axis=1) / box_width
                col_white_ratios = np.sum(extracted_ear_image == 255, axis=0) / box_height

                is_horizontal_line = np.mean(row_white_ratios > 0.9) > 0.8  # Most rows are almost fully white
                is_vertical_line = np.mean(col_white_ratios > 0.9) > 0.8  # Most columns are almost fully white

                # If it is a line, return False
                if is_horizontal_line or is_vertical_line:
                    print("This box is almost filled with white. Return False")
                    return False

        # Determine which top part to check based on ear position
        if ear_position == "left":
            top_part_to_check = top_right_half  # Check the right part for left ear
            other_top_part = top_left_half
        elif ear_position == "right":
            top_part_to_check = top_left_half  # Check the left part for right ear
            other_top_part = top_right_half
        else:
            raise ValueError("Invalid ear_position. Must be 'left' or 'right'.")

        # Calculate white pixel distributions relative to top_half
        total_top_pixels = top_half.size
        white_pixel_ratio_top_to_check = np.sum(top_part_to_check == 255) / total_top_pixels * 100
        white_pixel_ratio_other_top = np.sum(other_top_part == 255) / total_top_pixels * 100

        # Calculate white pixel distribution between top and bottom half
        total_box_pixels = extracted_ear_image.size
        white_pixel_top_half = np.sum(top_half == 255) / total_box_pixels * 100
        white_pixel_bottom_half = np.sum(bottom_half == 255) / total_box_pixels * 100

        # If different between top and bottom half is so big, return False
        if (white_pixel_bottom_half >= 10) and (white_pixel_top_half <= 3):
            print("Different between top and bottom half is so big. This is almost not an ear. Return False")
            return False
        else:
            print("Top half white distribution: " + str(white_pixel_top_half))
            print("Bottom half white distribution: " + str(white_pixel_bottom_half))

        if white_pixel_ratio_top_to_check == 0:
            print("White pixel distribution top to check is zero. Return False")
            return False

        print("White pixel percent top to check: " + str(white_pixel_ratio_top_to_check))
        print("White pixel percent other top: " + str(white_pixel_ratio_other_top))

        # Check if the top part to check has a higher white pixel distribution than the other top part
        # if (white_pixel_ratio_top_to_check >= 1) and (white_pixel_ratio_other_top >= 1):
        if white_pixel_ratio_top_to_check > 0.3:
            return True
        else:
            print("White pixel top to check of side " + ear_position + " is lower than 0.3. Return False")
            #return True
            return False

    def count_pins(self, contours, longest_box_data, height_offset=5, y_offset=5, target_pin_count=12):
        """
        Counts the number of bounding boxes (pins) based on heights within a range and groups
        the remaining boxes based on continuous y-coordinate values.

        Args:
            contours (List[ndarray]): List of contours to analyze.
            longest_box_data (Tuple[int, int, int, int]): The bounding box (x, y, w, h) of the longest contour.
            height_offset (int): Allowed height offset for grouping pin detection.
            y_offset (int): Allowed y-coordinate offset for filtering bounding boxes.
            target_pin_count (int): Desired count of pins to find in a group.

        Returns:
            Tuple[int, List[Tuple[int, int, int, int]]]: The count of detected pins and their bounding boxes.
        """
        longest_x, longest_y, longest_w, longest_h = longest_box_data
        filtered_bounding_boxes = []

        # Print bounding box data before processing
        # print("Bounding boxes data before processing:")
        for contour in contours:
            box = cv2.boundingRect(contour)
            # print(f"Box: {box}")  # Print the (x, y, w, h) of each bounding box
            filtered_bounding_boxes.append(box)

        # Print filtered bounding boxes data
        # print("Filtered bounding boxes data:")
        # for box in filtered_bounding_boxes:
        #    print(f"Filtered Box: {box}")  # Print the filtered bounding boxes

        # Remove boxes where width is greater than height
        filtered_bounding_boxes = [
            box for box in filtered_bounding_boxes if box[2] <= box[3]
        ]

        # Print remaining bounding boxes after removal
        # print("Remaining bounding boxes after removing w > h:")
        # for box in filtered_bounding_boxes:
        #    print(f"Remaining Box: {box}")

        # Determine the height range based on the filtered boxes
        if filtered_bounding_boxes:
            heights = [box[3] for box in filtered_bounding_boxes]  # Get heights of remaining boxes
            min_height = min(heights)
            max_height = max(heights)

            # Filter boxes based on height range
            filtered_bounding_boxes = [
                box for box in filtered_bounding_boxes
                if min_height - height_offset <= box[3] <= max_height + height_offset
            ]

            # Print bounding boxes after height filtering
            # print("Bounding boxes after height filtering:")
            # for box in filtered_bounding_boxes:
            #    print(f"Height Filtered Box: {box}")

            # Group boxes based on continuous y values
            filtered_bounding_boxes.sort(key=lambda b: b[1])  # Sort by y-coordinate
            groups = []
            current_group = [filtered_bounding_boxes[0]]

            for box in filtered_bounding_boxes[1:]:
                if abs(box[1] - current_group[-1][1]) <= y_offset:
                    current_group.append(box)
                else:
                    groups.append(current_group)
                    current_group = [box]

            # Add the last group
            if current_group:
                groups.append(current_group)

            # Remove boxes which are inside another box
            # cleaned_groups = self.clean_groups(groups)

            # Remove boxes which overlap others
            # overlap_removed_groups = self.filter_overlapping_boxes(cleaned_groups)

            # Remove boxes which has x value far away remaining ones
            # x_threshold = 10
            # final_filtered_groups = self.filter_far_away_boxes(cleaned_groups, x_threshold)

            # Use non-max suppression to remove overlapping boxes
            first_nms_groups = self.non_max_suppression(groups, overlap_threshold=0.2)

            # second_nms_groups = self.non_max_suppression(first_nms_groups, overlap_threshold=0.8)

            # third_nms_groups = self.non_max_suppression(second_nms_groups, overlap_threshold=0.1)

            # Check for a group that matches the target pin count
            matching_groups = [group for group in first_nms_groups if len(group) == target_pin_count]

            if matching_groups:
                largest_group = matching_groups[0]  # Take the first matching group
            else:
                if len(first_nms_groups) == target_pin_count + 1:
                    sorted_groups = self.non_max_suppression(first_nms_groups, overlap_threshold=0.001)
                else:
                    # Sort groups by number of elements (size)
                    sorted_groups = sorted(first_nms_groups, key=len, reverse=True)

                # Get the largest group by count and handle the case for groups with more than 2 with at least 5 elements
                if len(sorted_groups) > 1:
                    # Filter groups with at least 5 elements
                    eligible_groups = [group for group in sorted_groups if len(group) >= 1]
                    if len(eligible_groups) > 1:
                        largest_group = eligible_groups[0]  # Get the first-largest group
                    else:
                        largest_group = sorted_groups[0]  # If not enough eligible, take the largest
                else:
                    largest_group = sorted_groups[0] if sorted_groups else []

            # Print final groups
            # print("All groups of bounding boxes based on y values:")
            # for group in groups:
            #    print(f"Group: {[box for box in group]}")

        else:
            largest_group = []

        # The count of valid pin bounding boxes
        pin_count = len(largest_group)

        # Print final bounding boxes in the largest group
        # print("Largest group of bounding boxes:")
        # for box in largest_group:
        #    print(f"Largest Box: {box}")

        return pin_count, largest_group

    def show_binary_image_distribution(self, binary_image, part_name):
        """
        Measure and display the distribution of foreground (white) vs. background (black) pixels in a binary image.

        Args:
            binary_image (ndarray): Input binary image (values should be 0 and 255).
            part_name (str): Name of the part associated with the image for display purposes.

        Returns:
            tuple: A tuple containing the percentage of black and white pixels.
        """
        # Calculate the total number of pixels
        total_pixels = binary_image.size

        # Count the number of black (0) and white (255) pixels
        black_pixels = np.count_nonzero(binary_image == 0)
        white_pixels = np.count_nonzero(binary_image == 255)

        # Calculate the percentage
        black_percentage = (black_pixels / total_pixels) * 100
        white_percentage = (white_pixels / total_pixels) * 100

        # Display the distribution
        print(f"Distribution for {part_name}:")
        print(f"  Total Pixels: {total_pixels}")
        print(f"  Black Pixels (Background): {black_pixels} ({black_percentage:.2f}%)")
        print(f"  White Pixels (Foreground): {white_pixels} ({white_percentage:.2f}%)")

        # Visualize as a bar chart
        # plt.figure(figsize=(6, 4))
        # plt.bar(["Black (Background)", "White (Foreground)"], [black_pixels, white_pixels], color=['blue', 'green'])
        # plt.title(f"Pixel Distribution in Binary Image for {part_name}")
        # plt.ylabel("Pixel Count")
        # plt.show()

        # Return the percentages as a tuple
        return black_percentage, white_percentage

    def is_image_full_color(self, binary_image):
        """
        Checks if the given binary image is completely black or completely white.

        Parameters:
        - binary_image: Input binary image (as a NumPy array).

        Returns:
        - bool: True if the image is full black or full white, False if mixed.
        """
        # Ensure the image is a binary image (0s and 255s)
        # if not np.array_equal(np.unique(binary_image), [0, 255]):
        #    raise ValueError("The image is not a binary image.")

        # Count the number of white and black pixels
        white_pixel_count = np.sum(binary_image == 255)
        black_pixel_count = np.sum(binary_image == 0)

        # Get the total number of pixels
        total_pixels = binary_image.size

        # Check if the image is full black or full white
        if white_pixel_count == total_pixels or black_pixel_count == total_pixels:
            return True
        else:
            return False

    def check_connector_lock_defect(self, resized_image, expected_pin_count, expected_top_left_pixel_density,
                                    expected_top_right_pixel_density, left_offset, right_offset):
        """
        Process and evaluate part 1 - Connector - Pin count
        """
        # Extract parts from resized image
        extracted_connector_image_path, self.extracted_jack_image_path, extracted_connector_image, self.extracted_jack_image = self.extract_parts(
            resized_image, show_images=False)

        # Extract parts from resized adjusted histogram image
        hist_extracted_connector_image_path, self.hist_extracted_jack_image_path, hist_extracted_connector_image, self.hist_extracted_jack_image = self.extract_parts(
            self.adjusted_histogram_image, show_images=False, is_connector_saved=False)

        _extracted_connector_image = extracted_connector_image.copy()

        # Convert extracted connector image to binary with optional dilation
        connector_binary_image_path, connector_binary_image, connector_processed_image = self.convert_to_binary(
            cv2.imread(extracted_connector_image_path, cv2.IMREAD_GRAYSCALE),  # Ensure reading as grayscale
            part_name="connector",  # Specify the part name for the filename
            show_images=False,  # Set to True to display images
            apply_erosion=False,  # Disable erosion
            apply_dilation=True,  # Enable dilation
            dilation_iteration=3
        )

        # If converted image is full black or white, return False
        if self.is_image_full_color(connector_processed_image):
            return False

        # Add black border around processed image
        bordered_connector_image = self.add_black_border(connector_processed_image, border_size=5)
        if self.is_images_shown:
            cv2.imshow("Bordered Connector Image", bordered_connector_image)

        # Find edges from the bordered processed image
        bordered_connector_edges = self.find_edges(bordered_connector_image, show_image=False)

        # Get bordered image contours and their metadata
        bordered_connector_contours, bordered_connector_info = self.find_contours(bordered_connector_edges,
                                                                                  show_image=False)

        _bordered_connector_image = bordered_connector_image.copy()

        # Draw all bounding boxes
        self.draw_bounding_boxes(_extracted_connector_image, bordered_connector_contours, show_image=False)

        # Draw bounding box for the longest contour
        length, longest_box, boxed_connector_image = self.find_and_draw_longest_contour(
            bordered_connector_contours,
            _extracted_connector_image,
            color=(255, 0, 0),  # Blue color in BGR
            thickness=2
        )
        self.longest_box = longest_box

        # Optionally display the boxed image
        if self.is_images_shown:
            cv2.imshow("Longest Contour Boxed", _extracted_connector_image)
            cv2.waitKey(0)

        # print("Length of the longest contour:", length)
        # if longest_box is not None:
        #    print("Bounding box for the longest contour:", longest_box)
        # else:
        #    print("No bounding box drawn.")

        # print("Connector Contours Info:", connector_info)

        # Count pin
        # Limit where we find and count area of pins
        # x = 0
        # y = 60
        # height = 160
        # width = 640

        longest_x, longest_y, longest_w, longest_h = longest_box
        x = longest_x
        y = longest_y
        height = longest_h // 2 + 20
        width = longest_w

        raw_limited_connector_processed_image = connector_processed_image[y:y + height, x:x + width]
        drawn_pin_limited_connector_processed_image = _extracted_connector_image[y:y + height, x:x + width]
        limited_connector_processed_image = self.add_white_border(raw_limited_connector_processed_image)
        limited_image_width = limited_connector_processed_image.shape[1]
        x_start = limited_image_width // 2 - 100
        top_cut_from_limited_image = limited_connector_processed_image[0:20, x_start:x_start + 200]
        if self.is_images_shown:
            cv2.imshow("Top Cut", top_cut_from_limited_image)
            cv2.waitKey(0)

        # Display the final image with pins highlighted
        if self.is_images_shown:
            cv2.imshow("Limited connector processed image", limited_connector_processed_image)
            cv2.waitKey(0)

        # Find edges from processed image
        connector_edges = self.find_edges(limited_connector_processed_image, show_image=False)

        # Get contours and their metadata
        connector_contours, connector_info = self.find_contours(connector_edges, show_image=False)

        # Draw bounding boxes on dilation image
        # boxed_connector_image = self.draw_bounding_boxes(extracted_connector_image, connector_contours,
        #                                                 show_image=True)

        # Filter and draw bounding boxes inside the longest box
        modified_image, filtered_contours = self.filter_and_draw_bounding_boxes(
            connector_contours,
            longest_box,
            extracted_connector_image,  # Image with the longest contour box already drawn
            color=(0, 255, 0),  # Green color for other bounding boxes
            thickness=2
        )

        # Display the modified image with filtered bounding boxes
        # cv2.imshow("Filtered Bounding Boxes Inside Longest Box", modified_image)
        # cv2.waitKey(0)

        # Count pins based on height and y-coordinate grouping
        _pin_count, _pin_bounding_boxes = self.count_pins(
            connector_contours,
            longest_box,
            height_offset=5,  # Set the height offset as needed
            y_offset=5,  # Set the y-offset as needed
            target_pin_count=12
        )

        print("Number of detected pins:", _pin_count)
        print("Pin bounding boxes:", _pin_bounding_boxes)

        # Optionally, draw the pin bounding boxes on the image
        for box in _pin_bounding_boxes:
            x, y, w, h = box
            # cv2.rectangle(_extracted_connector_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for pins
            # self.pin_detected_image = _extracted_connector_image.copy()
            cv2.rectangle(drawn_pin_limited_connector_processed_image, (x, y), (x + w, y + h), (0, 255, 0),
                          thickness=2)  # Green color for pins
        self.pin_detected_image = drawn_pin_limited_connector_processed_image.copy()

        # Save pin detected image
        # Create a timestamped filename for the image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pin_detected_image_path = os.path.join(self.binary_images_directory_path,
                                                    f"pin_detected_binary_image_{timestamp}.png")

        # Save the binary image
        success = cv2.imwrite(self.pin_detected_image_path, self.pin_detected_image)
        if success:
            print(f"Pin detected image saved at: {self.pin_detected_image_path}")
        else:
            print("Failed to save the pin detected image.")
        # Display the final image with pins highlighted
        if self.is_images_shown:
            cv2.imshow("Pins Highlighted", _extracted_connector_image)
            cv2.waitKey(0)

        # For debugging only
        # if expected_pin_count == _pin_count - 1:
        #    _pin_count -= 1
        # return True

        if expected_pin_count != _pin_count:
            print("Pin count not match. Detected " + str(_pin_count) + " while expected " + str(expected_pin_count))
            return False

        """
        Process and evaluate part 1 - Connector - Shape correctness
        """
        # From dilated binary image, take longest_box's coordinates as reference, then from these coordinates,
        # crop top left and right corner as squares, size defined by crop_size
        # Count for number of white pixels in crop squares, print to debug
        # Dilated binary image variable name is connector_processed_image

        # Assuming longest_box is defined as (longest_x, longest_y, longest_w, longest_h)
        longest_x, longest_y, longest_w, longest_h = longest_box
        # pin_1_x, pin_1_y, pin_1_w, pin_1_h = _pin_bounding_boxes[0]
        # print("Pin 1 bounding box: x: " + str(pin_1_x) + ", y: " + str(pin_1_y))
        # pin_12_x, pin_12_y, pin_12_w, pin_12_h = _pin_bounding_boxes[len(_pin_bounding_boxes) - 1]
        # print("Pin 12 bounding box: x: " + str(pin_12_x) + ", y: " + str(pin_12_y))

        # Define the size of the crop square (can be changed as needed)
        crop_size = 50  # Change this value to adjust the crop size

        # Ensure the crop size does not exceed the dimensions of the image
        if crop_size > longest_w or crop_size > longest_h:
            raise ValueError("Crop size must be less than or equal to the dimensions of the bounding box.")

        # Crop top left corner
        top_left_x = longest_x
        top_left_y = longest_y
        # top_left_x = pin_1_x - 300
        # top_left_y = pin_1_y - 100
        top_left_crop = bordered_connector_image[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]
        # top_left_crop = bordered_connector_image[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]

        # Crop top right corner
        top_right_x = longest_x + longest_w - crop_size
        top_right_y = longest_y
        # top_right_x = pin_12_x
        # top_right_y = pin_12_y - 50
        top_right_crop = bordered_connector_image[top_right_y:top_right_y + crop_size,
                         top_right_x:top_right_x + crop_size]
        # top_right_crop = bordered_connector_image[top_right_y:top_right_y + crop_size,
        #                 top_right_x:top_right_x + crop_size]

        # Count white pixels in top left crop
        top_left_white_count = cv2.countNonZero(top_left_crop)
        print(f"White pixels in top left corner: {top_left_white_count}")

        # Count white pixels in top right crop
        top_right_white_count = cv2.countNonZero(top_right_crop)
        print(f"White pixels in top right corner: {top_right_white_count}")

        # Show the cropped squares
        if self.is_images_shown:
            cv2.imshow("Top Left Crop", top_left_crop)
            cv2.imshow("Top Right Crop", top_right_crop)

        # Wait for a key press and close the image windows
        cv2.waitKey(0)

        if top_left_white_count > expected_top_left_pixel_density + left_offset:
            print("Connector shape correctness check: NG - Left density over allowed")
            print("Expected: " + str(expected_top_left_pixel_density + left_offset))
            return False
        if top_right_white_count > expected_top_right_pixel_density + right_offset:
            print("Connector shape correctness check: NG - Right density over allowed")
            print("Expected: " + str(expected_top_right_pixel_density + right_offset))
            return False

        # Check if connector is opening at 90 degree
        if not self.is_image_full_color(top_cut_from_limited_image):
            print("Connector shape correctness check: NG - Connector opening at 90 degree")
            return False

        print("Connector shape correctness check: OK")
        return True

    def check_jack_fit_defect(self):
        """
        Erode to check if FPC is straight - White pixel distribution low
        """
        white_pixel_distribution_percent_threshold = 30

        # Convert to binary image with threshold = 60
        self.jack_binary_path, self.jack_binary_image, jack_processed_image = self.convert_to_binary_single_threshold(
            cv2.imread(self.extracted_jack_image_path),
            part_name="jack",
            threshold=65,
            show_images=False,
            apply_erosion=True,
            apply_dilation=True,
            apply_closing=False
        )

        # If converted image is full black or white, return False
        if self.is_image_full_color(self.jack_binary_image):
            return False

        # Check distribution of black-white pixel
        # Define the crop coordinates (x, y, width, height)
        x, y, width, height = 160, 100, 320, 140
        # x, y, width, height = 120, 20, 400, 460

        # Crop the area from jack_processed_image
        fpc_area_image = jack_processed_image[y:y + height, x:x + width]

        # Optionally, display the cropped image
        # cv2.imshow("FPC Area Image", fpc_area_image)
        # cv2.waitKey(0)

        black_pixel_percent, white_pixel_percent = self.show_binary_image_distribution(fpc_area_image, part_name="jack")

        if white_pixel_percent > white_pixel_distribution_percent_threshold:
            print("White pixel distribution exceeds threshold - NG")
            # return False
        else:
            print("White pixel distribution is in allowed range - OK")

        # If pixel percent is very low, we assume that FPC fit defect check is ok. No more check needed
        if white_pixel_percent <= 5:
            print("White pixel distribution is very low - Needn't to check more")
            # return True

        """
            Check left and right's white pixel difference of FPC cable to check if the cable is straight
        """

        # Crop fpc are image to 2 parts, left and right, then measure white pixel distribution of both and compare
        x_left, y_left, width_left, height_left = 0, 0, 170, 120
        x_right, y_right, width_right, height_right = 170, 0, 170, 120

        left_fpc_area = fpc_area_image[y_left:y_left + height_left, x_left:x_left + width_left]
        right_fpc_area = fpc_area_image[y_right:y_right + height_right, x_right:x_right + width_right]

        # cv2.imshow("Left FPC Area Image", left_fpc_area)
        # cv2.waitKey(0)

        # cv2.imshow("Right FPC Area Image", right_fpc_area)
        # cv2.waitKey(0)

        left_fpc_black_pixel_distribution, left_fpc_white_pixel_distribution = self.show_binary_image_distribution(
            left_fpc_area,
            part_name="left_fpc_area")
        right_fpc_black_pixel_distributio, right_fpc_white_pixel_distribution = self.show_binary_image_distribution(
            right_fpc_area,
            part_name="right_fpc_area")

        white_pixel_difference = abs(left_fpc_white_pixel_distribution - right_fpc_white_pixel_distribution)

        # Check white pixel distribution difference, standard is 2.5
        diff = 2
        if white_pixel_difference >= diff:
            print("White pixel distribution difference between left and right fpc area is over allowed: " + str(
                white_pixel_difference))
            # return False
        else:
            print("White pixel distribution difference between left and right fpc area is in allowed range: " + str(
                white_pixel_difference))
            # return True

        """
            Dilate to check if FPC's left and right ears are balanced
        """
        # Crop the left and right ears
        longest_box_x, longest_box_y, longest_box_w, longest_box_h = self.longest_box
        x_left, y_left, w_left, h_left = longest_box_x - 20, 30, 90, 120
        x_right, y_right, w_right, h_right = longest_box_x + 300, 30, 90, 120

        # Crop the area which contains both
        x_both, y_both, w_both, h_both = longest_box_x - 20, 30, 420, 250
        # both_area = self.hist_extracted_jack_image[y_both:y_both + h_both, x_both:x_both + w_both]

        # Correct the cropping logic
        left_ear = self.hist_extracted_jack_image[y_left:y_left + h_left, x_left:x_left + w_left]
        right_ear = self.hist_extracted_jack_image[y_right:y_right + h_right, x_right:x_right + w_right]

        # Binarize the left ear
        _left_ear_binary_path, _left_ear_binary_image, _left_ear_processed_image = self.convert_to_binary_single_threshold(
            left_ear,
            part_name="left_ear",
            use_adaptive_threshold=True,  # Enable adaptive thresholding
            # adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian method
            adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,  # Use Mean C method
            adaptive_block_size=21,  # Larger block size for more stable thresholding
            adaptive_c=5,  # Constant subtracted from mean
            show_images=True,
            apply_erosion=True,
            apply_dilation=True,
            apply_closing=False,
            apply_opening=False,
            erosion_iteration=2,
            dilation_iteration=1,
            is_binary_invert=True  # Keep the inversion since we want to detect dark objects
        )
        _left_ear_processed_image = cv2.cvtColor(_left_ear_processed_image, cv2.COLOR_GRAY2BGR)

        # Binarize the right ear
        _right_ear_binary_path, _right_ear_binary_image, _right_ear_processed_image = self.convert_to_binary_single_threshold(
            right_ear,
            part_name="right_ear",
            use_adaptive_threshold=True,  # Enable adaptive thresholding
            # adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian method
            adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,  # Use Mean C method
            adaptive_block_size=21,  # Larger block size for more stable thresholding
            adaptive_c=5,  # Constant subtracted from mean
            show_images=False,
            apply_erosion=True,
            apply_dilation=False,
            apply_closing=True,
            apply_opening=False,
            erosion_iteration=2,
            dilation_iteration=1,
            is_binary_invert=True  # Keep the inversion since we want to detect dark objects
        )
        _right_ear_processed_image = cv2.cvtColor(_right_ear_processed_image, cv2.COLOR_GRAY2BGR)

        # Optionally, display the cropped image
        # cv2.imshow("Left Ear", left_ear)
        # cv2.waitKey(0)
        # Optionally, display the cropped image
        # cv2.imshow("Right Ear", right_ear)
        # cv2.waitKey(0)
        # Optionally, display the cropped image

        # Add black border to both ears crop image
        # bordered_both_area = self.add_black_border(both_area)
        # if self.is_images_shown:
        #    cv2.imshow("Bordered Both Area", bordered_both_area)
        #    self.draw_min_area_rect(bordered_both_area, show_image=True)
        #    cv2.waitKey(0)
        bordered_left_ear = self.add_black_border(_left_ear_processed_image)
        bordered_right_ear = self.add_black_border(_right_ear_processed_image)
        _bordered_left_ear = bordered_left_ear.copy()
        _bordered_right_ear = bordered_right_ear.copy()
        # cv2.imshow("Bordered Both Are", bordered_both_area)
        # cv2.waitKey(0)

        # Find edges from processed image
        left_ear_edges = self.find_edges(bordered_left_ear, show_image=False)
        right_ear_edges = self.find_edges(bordered_right_ear, show_image=False)

        # Get contours and their metadata
        left_ear_contours, left_ear_info = self.find_contours(left_ear_edges, show_image=False)
        right_ear_contours, right_ear_info = self.find_contours(right_ear_edges, show_image=False)

        # Extract bounding boxes and append to a list
        left_ear_bounding_boxes = []
        for contour in left_ear_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= h:
                left_ear_bounding_boxes.append((x, y, w, h))

        self.draw_ear_bounding_boxes(_bordered_left_ear, left_ear_bounding_boxes, name="left")

        right_ear_bounding_boxes = []
        for contour in right_ear_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= h:
                right_ear_bounding_boxes.append((x, y, w, h))

        self.draw_ear_bounding_boxes(_bordered_right_ear, right_ear_bounding_boxes, name="right")

        # Sort the bounding boxes by height in descending order
        # left_ear_bounding_boxes.sort(key=lambda box: box[3], reverse=True)  # box[3] is height
        # right_ear_bounding_boxes.sort(key=lambda box: box[3], reverse=True)  # box[3] is height
        left_ear_bounding_boxes.sort(key=lambda box: (-box[3], box[1]))  # box[3] is height, box[1] is y value
        right_ear_bounding_boxes.sort(key=lambda box: (-box[3], box[1]))  # box[3] is height, box[1] is y value

        # Get the first highest boxes
        if len(left_ear_bounding_boxes) > 0:
            # Analyze left ear (Γ shape)
            best_left_index, best_left_score, annotated_left = self.ear_analyzer.analyze_boxes(
                binary_image=bordered_left_ear,
                boxes=left_ear_bounding_boxes,
                contours=left_ear_contours,
                ear_position="left",
                show_results=True
            )

            if best_left_score >= self.ear_analyzer.min_score_threshold:
                box1 = left_ear_bounding_boxes[best_left_index]
                print(f"Found valid left ear pattern (score: {best_left_score:.2f})")
                cv2.imshow("Left ear analysis", annotated_left)
                cv2.waitKey(0)
            else:
                print("Not found any valid left ear pattern box. Return False")
                return False
        else:
            print("No box detected for left ear. Return False")
            return False

        if len(left_ear_bounding_boxes) > 0:
            # Analyze left ear (Γ shape)
            best_right_index, best_right_score, annotated_right = self.ear_analyzer.analyze_boxes(
                binary_image=bordered_right_ear,
                boxes=right_ear_bounding_boxes,
                contours=right_ear_contours,
                ear_position="right",
                show_results=True
            )

            if best_right_score >= self.ear_analyzer.min_score_threshold:
                box2 = right_ear_bounding_boxes[best_right_index]
                print(f"Found valid right ear pattern (score: {best_right_score:.2f})")
                cv2.imshow("Right ear analysis", annotated_right)
                cv2.waitKey(0)
            else:
                print("Not found any valid right ear pattern box. Return False")
                return False
        else:
            print("No box detected for right ear. Return False")
            return False

        # Draw bounding boxes on dilation image
        left_ear_boxed_image = self.draw_bounding_box(bordered_left_ear, box1)
        right_ear_boxed_image = self.draw_bounding_box(bordered_right_ear, box2)
        if self.is_images_shown:
            cv2.imshow("Left Ear Box", left_ear_boxed_image)
            cv2.waitKey(0)
            cv2.imshow("Right Ear Box", right_ear_boxed_image)
            cv2.waitKey(0)

        # Draw detected ear on the image
        self.temp_hist_extracted_jack_image = self.hist_extracted_jack_image.copy()
        self.temp_hist_extracted_jack_image = cv2.cvtColor(self.temp_hist_extracted_jack_image, cv2.COLOR_GRAY2BGR)
        self._extracted_jack_image = self.add_black_border(self.temp_hist_extracted_jack_image)
        # Draw rectangles for box1
        cv2.rectangle(
            self._extracted_jack_image,
            (box1[0] + x_left + 2, box1[1] + y_left + 2),  # Top-left corner
            (box1[0] + box1[2] + x_left, box1[1] + box1[3] + y_left),  # Bottom-right corner
            (0, 255, 0),  # Green color in BGR
            thickness=2
        )

        # Draw rectangles for box2
        cv2.rectangle(
            self._extracted_jack_image,
            (box2[0] + x_right + 2, box2[1] + y_right + 2),  # Top-left corner
            (box2[0] + box2[2] + x_right, box2[1] + box2[3] + y_right),  # Bottom-right corner
            (0, 255, 0),  # Green color in BGR
            thickness=2
        )

        if self.is_images_shown:
            cv2.imshow("Finale", self._extracted_jack_image)
            cv2.waitKey(0)

        # Save final jack image
        cv2.imwrite(self.extracted_jack_image_path, self._extracted_jack_image)

        # Compare the y-values of the two boxes
        y_difference = abs(box1[1] - box2[1])  # box[1] is the y-coordinate
        print("Left and right ears difference: " + str(y_difference))
        offset = 7  # Set your desired offset value

        # Print NG or OK based on the y-value comparison
        if y_difference <= offset:
            print("Left and right ears balance check - OK")
            return True
        else:
            print("Left and right ears balance check - NG")
            return False

    def stop(self):
        """Stop the processing thread."""
        self.is_running = False


if __name__ == "__main__":
    image_processor = ImageProcessor()

    # Set config dict
    image_processor.config_dict = {
        "part-name": "Connector - Jack",
        "raw-images-directory-path": r"C:\BoardDefectChecker\images\raw-images",
        "resized-images-directory-path": r"C:\BoardDefectChecker\images\resized-images",
        "binary-images-directory-path": r"C:\BoardDefectChecker\images\binary-images",
        "edge-images-directory-path": r"C:\BoardDefectChecker\images\edge-images",
        "ng-images-directory-path": r"C:\BoardDefectChecker\images\ng-images",
        "ok-images-directory-path": r"C:\BoardDefectChecker\images\ok-images",
        "connection-pin-count": 10,
        "component-1-roi-coordinates": {
            "top-left-x": 0,
            "top-left-y": 0,
            "bottom-right-x": 640,
            "bottom-right-y": 240
        },
        "component-2-roi-coordinates": {
            "top-left-x": 0,
            "top-left-y": 240,
            "bottom-right-x": 640,
            "bottom-right-y": 480
        },
        "pixel-density-edge-1": 150,
        "pixel-density-edge-2": 53,
        "pixel-density-edge-3": 34,
        "tilt-angle-edge-1-2": 60,
        "tilt-angle-edge-2-3": 60
    }

    # Directory paths for saving images
    image_processor.raw_images_directory_path = r"C:\BoardDefectChecker\images\raw-images"
    image_processor.resized_images_directory_path = r"C:\BoardDefectChecker\images\resized-images"
    image_processor.binary_images_directory_path = r"C:\BoardDefectChecker\images\binary-images"
    image_processor.edge_images_directory_path = r"C:\BoardDefectChecker\images\edge-images"
    image_processor.ng_images_directory_path = r"C:\BoardDefectChecker\images\ng-images"
    image_processor.ok_images_directory_path = r"C:\BoardDefectChecker\images\ok-images"

    # Set image path to test
    # image_path = r"D:\Working\Images\Grayscale\Sample-3\BoardCheckSample-2.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-3\BoardCheckSample-1.bmp"
    # 180 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-1.bmp"
    # 90 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-2.bmp"
    # 45 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-3.bmp"
    # 30 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-4.bmp"
    # Image from camera on JIG - 180 degree - With light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-6.bmp"
    # Image from camera on JIG - 180 degree - Without light 1
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-2.bmp"
    # Image from camera on JIG - 180 degree - Without light 2
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-7.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-8.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-9.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-10.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-11.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-12.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-13.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-14.bmp"
    # image_path = r"C:\Work\Sharp\Images\Grayscale\Sample-5\Sample-5-08.bmp"
    image_path = r"C:\Work\Sharp\Images\Grayscale\Sample-6\Sample-6-02.bmp"
    # Image from camera on JIG - 90 degree - Without light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-3.bmp"
    # Image from camera on JIG - 45 degree - Without light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-10.bmp"
    # Image from camera on JIG - 30 degree - Without light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-9.bmp"

    image_processor.set_image_path(image_path)

    image_processor.is_images_shown = True

    # Preprocessing by resizing to 640x480
    # resized_image_path, resized_image = image_processor.preprocessing(image_path, show_image=True)
    resized_image_path, resized_image = image_processor.preprocessing(
        image_path,
        show_image=False,
        apply_median_blur=True,
        median_kernel_size=7,  # Larger kernel for stronger noise reduction
        apply_gaussian_blur=True,
        gaussian_kernel_size=5,  # Larger kernel for more smoothing
        gaussian_sigma=1  # Larger sigma for stronger blur
    )

    # Analyze histogram
    image_processor.analyze_histogram(resized_image_path, show_plot=False)

    # Adjust histogram
    image_processor.adjust_histogram(resized_image_path, show_result=False)

    """
    Process and evaluate part 1 - Connector
    """
    # Check pin count
    if image_processor.check_connector_lock_defect(resized_image, 12, 1563, 1600, 225, 95):
        print("Connector Lock Defect Check: OK")
    else:
        print("Connector Lock Defect Check: NG")

    """
       Process and evaluate part 2 - Jack (FPC Lead)
    """
    image_processor.check_jack_fit_defect()
