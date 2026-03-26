
def are_same(square1, square2, threshold=0.33):
    intersection = square1.intersection(square2)

    if (
        ((square1.area - intersection.area) / square1.area) < threshold
        or ((square2.area - intersection.area) / square2.area) < threshold
    ):

        return True

    return False


def is_walking(current_square, previous_square):
    return current_square.centroid.distance(previous_square.centroid) > 5


def width(square):
    minx, miny, maxx, maxy = square.bounds

    return maxx - minx


def close_each_other(square1, square2):
    square1_width = width(square1)
    square2_width = width(square2)

    distance = square1.centroid.distance(square2.centroid)

    return distance <= (square1_width + square2_width) / 2
