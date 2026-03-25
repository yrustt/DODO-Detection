
def are_same(square1, square2, threshold=0.33):
    intersection = square1.intersection(square2)

    if (
        ((square1.area - intersection.area) / square1.area) < threshold
        or ((square2.area - intersection.area) / square2.area) < threshold
    ):

        return True

    return False
