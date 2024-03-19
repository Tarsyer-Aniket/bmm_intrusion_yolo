import streamlit as st

def point_side_of_line(line, point):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = point
    # Vector AB
    AB_x = x2 - x1
    AB_y = y2 - y1

    # Vector AP
    AP_x = x - x1
    AP_y = y - y1

    # Calculate cross product
    cross_product = (AB_x * AP_y) - (AB_y * AP_x)
    st.info(cross_product)
    # Interpret the result
    if cross_product < 0:
        return "Store is on the left side of the line"
    elif cross_product > 0:
        return "Store is on the right side of the line"
    else:
        return "Store is on the line"



# assumes line segments are stored in the format [(x0,y0),(x1,y1)]
def intersects(s0,s1):
    dx0 = s0[1][0]-s0[0][0]
    dx1 = s1[1][0]-s1[0][0]
    dy0 = s0[1][1]-s0[0][1]
    dy1 = s1[1][1]-s1[0][1]
    p0 = dy1*(s1[1][0]-s0[0][0]) - dx1*(s1[1][1]-s0[0][1])
    p1 = dy1*(s1[1][0]-s0[1][0]) - dx1*(s1[1][1]-s0[1][1])
    p2 = dy0*(s0[1][0]-s1[0][0]) - dx0*(s0[1][1]-s1[0][1])
    p3 = dy0*(s0[1][0]-s1[1][0]) - dx0*(s0[1][1]-s1[1][1])
    return (p0*p1<=0) & (p2*p3<=0)


def check_crossed_line_direction(line, line2):

    x1, y1 = line[0]
    x2, y2 = line[1]

    prev_pos = line2[0]
    curr_pos = line2[1]

    
    # Check if a person has crossed a line defined by two points (x1, y1) and (x2, y2) in a particular direction.
    
    # :param x1, y1: Coordinates of the first point of the line
    # :param x2, y2: Coordinates of the second point of the line
    # :param prev_pos: The previous position of the person as a tuple (x, y)
    # :param curr_pos: The current position of the person as a tuple (x, y)
    # :return: 'crossed' if the line was crossed, 'same_side' if still on the same side, or 'no_cross' if on the opposite side without crossing
    

    def position_to_line(px, py):
        # Returns a tuple with the determinant and its sign
        det = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        return det, np.sign(det)

    prev_det, prev_sign = position_to_line(*prev_pos)
    curr_det, curr_sign = position_to_line(*curr_pos)

    # Check if the line was crossed by looking at the sign change
    if prev_sign != curr_sign:
        return 'crossed'
    elif prev_sign == curr_sign and curr_det == 0:
        # The person is exactly on the line at the current position
        return 'no_cross'
    else:
        # The person is still on the same side of the line
        return 'same_side'

def find_radius_endpoint(center_x, center_y, radius, angle_in_degrees):
  """Calculates the coordinates of the radius vector endpoint."""

  angle_in_radians = angle_in_degrees * math.pi / 180  # Convert angle to radians
  radius_end_x = center_x + radius * math.cos(angle_in_radians)
  radius_end_y = center_y + radius * math.sin(angle_in_radians)

  return radius_end_x, radius_end_y
