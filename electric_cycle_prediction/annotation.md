The HNU Dataset consists of annotated videos of cycles, electric cycles, pedestrians, vehicles, which been collected in HUNAN University.

This HNU Dataset contains seven files: cycle, electric cycle, pedestrian, vehicles, example.jpg, HNU.csv, HNU.mp4.

We annotate the trajectory every 5 frames.
The fps of the HNU.mp4 is 25.

Annotation file format(cycle, electric_cycle, pedestrian, vehicle):
The name of the annotations.txt corresponds to the ID.
Each row in the annotations.txt file corresponds to an annotation.

##The definition of these rows are (cycle, electric_cycle, pedestrian):

    1+n*3   frame. The number of frames in which the ID appears.
    2+n*3   x. The x-coordinate of the ID(image-coordinate).
    3+n*3   y. The y-coordinate of the ID(image-coordinate).
   attention: The last row doesn't mean anything.

###The definition of these rows are (vehicle):

    1+n*9   frame. The number of frames in which the ID appears.
    2+n*9   x. The x-coordinate of the first point(image-coordinate).
    3+n*9   y. The y-coordinate of the first point(image-coordinate).
    4+n*9   x. The x-coordinate of the second point(image-coordinate).
    5+n*9   y. The y-coordinate of the second point(image-coordinate).
    6+n*9   x. The x-coordinate of the third point(image-coordinate).
    7+n*9   y. The y-coordinate of the third point(image-coordinate).
    8+n*9   x. The x-coordinate of the fourth point(image-coordinate).
    9+n*9   y. The y-coordinate of the fourth point(image-coordinate).
   attention: The last row doesn't mean anything.


##The file of HNU.csv:

    The first row represents frame;
    The second row represents ID;
    The third row represents y-coordinate(image-coordinate);
    The fourth row represents x-coordinate(image-coordinate);
    The last row represents the attribute(1:cycle, 2:electric-cycle, 3:pedestrian, 4:vehicle).

##There are four image-coordinates and four world-coordinates, and they correspond to each other (you can get homography matrix by opencv function):

    image-coordinates:[656,494],[836,485],[695,961],[1014,952].
    world-coordinates:[3000,4000],[3428,4000],[3203.85,5432.57], [3763.52,5451.73].

