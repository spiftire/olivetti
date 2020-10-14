import matplotlib.pyplot as plt


def showPicturePlot(picSize, pictureArray):
    plt.figure()
    rows = pictureArray.shape[0]
    reduced_pic = [None] * rows
    for i in range(rows):
        reduced_pic[i] = pictureArray[i].reshape(picSize, picSize)
    print(reduced_pic[0].shape)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(reduced_pic[i])
    plt.show()