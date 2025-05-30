import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sklearn.mixture
import math
import json

# defining surface and axes
# partition the data of under 18 and over 18
n=100
datas = ['comix1']
for i, data in enumerate(datas):
    with open(f'input_data/gmm/optimal_components_{data}_log.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}.json', 'r') as f:
        egos_pre = json.load(f)
    # reduce egos to 2 dimensions 
    egos = [{'age': 0 if ego['age'] < 2 else 1, 'contacts': [sum(ego['contacts'][:3]), sum(ego['contacts'][3:])], 'degree': ego['degree']} for ego in egos_pre]
    classifier = []
    samples = []
    for l in [0,1]:
        classifier.append(sklearn.mixture.GaussianMixture(n_components=10, covariance_type='full'))
        egos_age = [a for a in egos if a['age'] == l]
        print(len(egos_pre), len(egos_age))
        ## use log(k+1) instead of k to fit
        X = [[math.log(b+1) for b in a['contacts']] for a in egos_age]
        classifier[l].fit(X)
    
        x=np.outer(np.linspace(0,7,n), np.ones(n)) 
        y=x.copy().T
        z=np.zeros((len(x), len(x[0])))
        for i in range(len(z)):
            tmp = classifier[l].score_samples([[x[i,j], y[i,j]] for j in range(len(x[i]))])
            # print(tmp)
            for j, val in enumerate(tmp):
                z[i,j] = val
        # for i in range(len(x)):
        #     for j in range(len(x[0])):
        #         z[i,j] = classifier[l].predict_proba(np.array(x[i][j],y[i][j]))
    
        # x = np.outer(np.linspace(-2, 2, 10), np.ones(10))
        # y = x.copy().T
        # z = np.cos(x ** 2 + y ** 3)
        # print(z)

        fig = plt.figure(figsize=(12,10))

        # syntax for 3-D plotting
        # ax = plt.axes(projection='3d')
        ax = fig.add_subplot(projection='3d')

        # syntax for plotting
        ax.plot_surface(x, y, z, cmap='viridis',
                        edgecolor='none')
        # ax.set_title(f'{l}')
        # plt.show()
        
        # Rotate the axes and update
        for angle in range(0, 360*4 + 1):
            # Normalize the angle to the range [-180, 180] for display
            angle_norm = (angle + 180) % 360 - 180

            # Cycle through a full rotation of elevation, then azimuth, roll, and all
            elev = azim = roll = 0
            if angle <= 360:
                elev = angle_norm
            elif angle <= 360*2:
                azim = angle_norm
            elif angle <= 360*3:
                roll = angle_norm
            else:
                elev = azim = roll = angle_norm

            # Update the axis view and title
            ax.view_init(elev, azim, roll)
            plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

            plt.draw()
            plt.pause(.001)