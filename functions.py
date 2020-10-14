import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def show_picture_plot(pic_size: int, picture_vector, ax: plt.Axes):
    # a = fig.get_axes()
    # print(a)
    # Setting new figure
    # fig = plt.figure(constrained_layout=True)

    # Setting up gridspec
    ncols = 3
    nrows = 3
    gridspec.GridSpec(ncols=ncols, nrows=nrows)
    rows = picture_vector.shape[0]

    # Initialize the array
    picture = arranging_picture_vector_as_picture_array(pic_size, picture_vector, rows)
    for col in range(ncols):
        for row in range(nrows):
            fax = ax.add_subplot(gridspec[col, row])

            # ax.subplot(330 + 1 + i)
            fax.imshow(picture[col + row])


def arranging_picture_vector_as_picture_array(pic_size, picture_vector):
    """Takes a picture vector and reshapes it into a square picture array based on pic size"""
    picture_array = picture_vector.reshape(pic_size, pic_size)
    return picture_array
