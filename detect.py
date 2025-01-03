import cv2
from ultralytics import YOLO
import os

def main():
    # Path to the exported NCNN model and input image
    model_path = "best_ncnn_model"
    image_path = "test2/tstimgg5.jpg"

    # Paths to save the images
    save_dir = "dimages"
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    original_image_path = os.path.join(save_dir, "original_image.jpg")
    result_image_path = os.path.join(save_dir, "result_image.jpg")
    cropped_image_path = os.path.join(save_dir, "cropped_image.jpg")

    # Initialize the YOLO model
    ncnn_model = YOLO(model_path)

    # Run inference on the input image
    results = ncnn_model(image_path)

    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Save the original image
    cv2.imwrite(original_image_path, original_image)
    print(f"Original image saved to: {original_image_path}")

    # Create a copy of the image for drawing the result
    result_image = original_image.copy()

    # Variables to store the bounding box with the highest confidence score
    best_box = None
    best_score = 0
    best_label = -1

    # Find the bounding box with the highest score
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
        scores = result.boxes.conf.numpy()  # Confidence scores
        labels = result.boxes.cls.numpy()  # Class labels (if available)

        for i, box in enumerate(boxes):
            score = scores[i]
            label = int(labels[i]) if labels is not None else -1

            # Update if the current box has a higher score
            if score > best_score:
                best_box = box
                best_score = score
                best_label = label

    # Draw the highest confidence bounding box and crop the image
    cropped_image = None
    if best_box is not None:
        x_min, y_min, x_max, y_max = map(int, best_box)  # Convert to integers

        # Ensure bounding box is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(original_image.shape[1], x_max)
        y_max = min(original_image.shape[0], y_max)

        # Draw bounding box with increased thickness
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=8)

        # Add label and confidence score with larger font size
        text = f"baby: {best_score:.2f}"  # Example: "baby: 0.95"
        font_scale = 1.5  # Larger font size
        font_thickness = 4  # Bolder text thickness

        # Calculate text size and position
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x_min
        text_y = y_min - 20  # Move the text slightly higher
        if text_y < 0:  # Adjust if text goes out of image bounds
            text_y = y_min + text_size[1] + 10

        # Draw text background and label with more padding
        cv2.rectangle(result_image, (text_x, text_y - text_size[1] - 15),
                      (text_x + text_size[0] + 15, text_y + 15), (0, 255, 0), -1)
        cv2.putText(result_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Crop the image to the bounding box
        cropped_image = original_image[y_min:y_max, x_min:x_max]

        # Save the cropped image
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f"Cropped image saved to: {cropped_image_path}")
    else:
        print("No bounding box detected.")  # Log the message

    # Save the result image
    cv2.imwrite(result_image_path, result_image)
    print(f"Result image saved to: {result_image_path}")

    # Display the images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Result Image", result_image)
    if cropped_image is not None:
        cv2.imshow("Cropped Image", cropped_image)
    else:
        print("No cropped image to display.")

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
