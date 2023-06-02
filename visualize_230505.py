
import numpy as np
import matplotlib.pyplot as plt

def loss_plot(result, k_fold, h_dim, num_layers, num_epoch, learning_rate, seed, title = "loss_plot") : 
    train_loss = np.asarray(result[1]).mean(axis = 0)
    test_loss = np.asarray(result[2]).mean(axis = 0)

    fig = plt.figure(figsize = (9, 6))

    ax = fig.add_subplot(2,3,1)
    for ind in range(k_fold) : 
        plt.plot(range(len(result[1][ind])), result[1][ind])
    plt.xlabel("Iterations")
    plt.title("Train loss")

    ax = fig.add_subplot(2,3,2)
    for ind in range(k_fold) : 
        plt.plot(range(len(result[1][ind])), result[1][ind])
    plt.xlabel("Iterations")
    plt.title("Train loss near 0")
    plt.ylim(0, 500)

    ax = fig.add_subplot(2,3,3)
    plt.plot(range(train_loss.shape[0]), train_loss)
    plt.xlabel("Iterations")
    plt.title("Train loss")
    plt.ylim(0, 500)

    ax = fig.add_subplot(2,3,4)
    for ind in range(k_fold) : 
        plt.plot(range(num_epoch), result[2][ind])
    plt.xlabel("Epochs")
    plt.title("Test loss")

    ax = fig.add_subplot(2,3,5)
    for ind in range(k_fold) : 
        plt.plot(range(num_epoch), result[2][ind])
    plt.xlabel("Epochs")
    plt.title("Test loss near 0")
    plt.ylim(0, 500)

    ax = fig.add_subplot(2,3,6)
    plt.plot(range(test_loss.shape[0]), test_loss)
    plt.xlabel("Epochs")
    plt.title("Test loss")
    plt.ylim(0, 500)

    fig.savefig(f'{title}/h{h_dim}_layer{num_layers}_lr{learning_rate}_SEED{seed}.png')
    
    return fig

def loss_plot2(result, k_fold, h_dim, num_layers, num_epoch, learning_rate, seed, title = "loss_plot") : 
    train_loss = np.asarray(result[1]).mean(axis = 0)
    test_loss = np.asarray(result[2]).mean(axis = 0)

    fig = plt.figure(figsize = (9, 6))

    ax = fig.add_subplot(2,2,1)
    for ind in range(k_fold) : 
        plt.plot(range(len(result[1][ind])), result[1][ind])
    # plt.xlabel("Iterations")
    plt.title("Train loss")

    ax = fig.add_subplot(2,2,2)
    for ind in range(k_fold) : 
        plt.plot(range(len(result[1][ind])), result[1][ind])
    # plt.xlabel("Iterations")
    plt.title("Train loss near 0")
    plt.ylim(0, 1)

    ax = fig.add_subplot(2,2,3)
    for ind in range(k_fold) : 
        plt.plot(range(num_epoch), result[2][ind])
    # plt.xlabel("Epochs")
    plt.title("Test loss")

    ax = fig.add_subplot(2,2,4)
    for ind in range(k_fold) : 
        plt.plot(range(num_epoch), result[2][ind])
    # plt.xlabel("Epochs")
    plt.title("Test loss near 0")
    plt.ylim(0, 1)

    fig.savefig(f'{title}/h{h_dim}_layer{num_layers}_lr{learning_rate}_SEED{seed}.png')
    
    return fig




def binary_plot(normal_age, patient_age, normal_pred, patient_pred, title = "temp.png") : 
    x, y = normal_age, normal_pred
    z, w = patient_age, patient_pred

    fig, ax = plt.subplots(figsize = (12, 9))
    ax.scatter(x, y, s=10, zorder=5, label = 'Normal group')
    ax.scatter(z, w, s=70, zorder=20, label = 'Patient group', marker = "+")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel("Age")
    plt.ylabel("Prediction")
    plt.legend()
    plt.savefig(title)

    return fig

def ternary_plot(normal_age, patient_age, normal_pred, patient_pred, title = "temp.png") : 
    x, y = normal_age, normal_pred
    z, w = patient_age, patient_pred


    fig, ax = plt.subplots(figsize = (12, 9))
    ax.scatter(x, y, s=10, zorder=5, label = 'Normal group')
    ax.scatter(z[0:48], w[0:48], s=70, zorder=20, label = 'MCI group', marker = "+")
    ax.scatter(z[48:80], w[48:80], s=70, zorder=20, label = 'AD group', marker = "x")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel("Age")
    plt.ylabel("Prediction")
    plt.legend()
    plt.savefig(title)

    return fig

def quaternary_plot(normal_age, patient_age, normal_pred, patient_pred, title = "temp.png") : 
    x, y = normal_age, normal_pred
    z, w = patient_age, patient_pred

    fig, ax = plt.subplots(figsize = (12,9))
    ax.scatter(x, y, s=10, zorder=5, label = 'Normal')
    ax.scatter(z[0:26], w[0:26], s=70, zorder=20, label = 'MCI-NC', marker = "+")
    ax.scatter(z[26:48], w[26:48], s=70, zorder=20, label = 'MCI-C', marker = "+")
    ax.scatter(z[48:80], w[48:80], s=70, zorder=20, label = 'AD', marker = "x")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel("Age")
    plt.ylabel("Prediction")
    plt.legend()
    plt.savefig(title)

    return fig