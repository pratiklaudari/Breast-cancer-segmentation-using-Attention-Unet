#if we wnat to keep preview part:
    # Open the uploaded image and segmentation result using PIL
    uploaded_image = Image.open(image_path)
    segmentation_result = Image.open(segmentation_result_path)

    # Preview the uploaded image and segmentation result
    preview_path = 'preview.png'
    combined = np.hstack((np.array(uploaded_image), np.array(segmentation_result)))
    Image.fromarray((combined * 255).astype(np.uint8)).save(preview_path)

    return jsonify({'segmentation_result_path': segmentation_result_path, 'preview_path': preview_path, 'predicted_class': predicted_class})
