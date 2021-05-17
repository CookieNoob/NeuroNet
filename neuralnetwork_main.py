import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt

import neuralnetwork_gui as ng
import pygame


def main():
    with np.load('mnist.npz') as data:
        images = data['training_images']
        labels = data['training_labels']

    layer_sizes = (784, 16, 16, 10)

    training_set_size = 30000

    trainingImages = images[:training_set_size]
    trainingLabels = labels[:training_set_size]

    testImages = images[training_set_size:]
    testlabels = labels[training_set_size:]

    plt.imshow(images[0].reshape(28,28),cmap='gray')
    for i in range(10):
        if labels[0][i] == 1:
            print("Die Zahl hier ist eine", i)
            break
    plt.show()


    Network = nn.NeuralNetwork(layer_sizes)
    Network.accuracy(testImages, testlabels)
    
    Network.train(trainingImages, trainingLabels, 2, 10, 4.0)
    Network.accuracy(testImages, testlabels)

    Network.train(trainingImages, trainingLabels, 8, 20, 2.0)
    Network.accuracy(testImages, testlabels)


    predict(Network, testImages[19])
    plt.imshow(testImages[19].reshape(28,28),cmap='gray')
    plt.show()

    eventloop(Network)



def predict(Network, field):
    prediction = Network.feedforward(field)
    for i in range(10):
        if prediction[i] == max(prediction):
            print("Diese Zahl ist eine", i)





def eventloop(Network):
    gui = ng.gui(400, 400)
    running = True
    drawing = False
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key ==pygame.K_SPACE:
                    gui.resetGrid()
                if event.key == pygame.K_RETURN:
                    predict(Network, gui.onedimList())
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    gui.paint(pygame.mouse.get_pos())
            elif event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()


