import cv2


def is_point_within_polygon(quadrilateral, point):
    # required structure of that ROI will be like
    # roi = np.array([[], [], [], []], np.int32)
    # perform pointPolygonTest
    pts = quadrilateral.reshape((-1, 1, 2))
    dist = cv2.pointPolygonTest(pts, point, False)
    is_within = dist > 0
    # Return the result
    return is_within