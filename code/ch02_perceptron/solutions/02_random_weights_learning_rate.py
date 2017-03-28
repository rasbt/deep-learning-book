randw_params_1 = perceptron_train(X_train, y_train, learning_rate=1.0,
                                  mparams=None, zero_weights=False)

for _ in range(2):
    _ = perceptron_train(X_train, y_train, mparams=randw_params_1)

x_min_1 = -2
y_min_1 = (-(randw_params_1['weights'][0] * x_min) /
           randw_params_1['weights'][1] -
           (randw_params_1['bias'] / randw_params_1['weights'][1]))

x_max_1 = 2
y_max_1 = (-(randw_params_1['weights'][0] * x_max) /
           randw_params_1['weights'][1] -
           (randw_params_1['bias'] / randw_params_1['weights'][1]))


randw_params_2 = perceptron_train(X_train, y_train, learning_rate=0.01,
                                  mparams=None, zero_weights=False)

for _ in range(2):
    _ = perceptron_train(X_train, y_train, mparams=randw_params_2)

x_min_2 = -2
y_min_2 = (-(randw_params_2['weights'][0] * x_min) /
           randw_params_2['weights'][1] -
            (randw_params_2['bias'] / randw_params_2['weights'][1]))

x_max_2 = 2
y_max_2 = (-(randw_params_2['weights'][0] * x_max) /
           randw_params_2['weights'][1] -
            (randw_params_2['bias'] / randw_params_2['weights'][1]))


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min_1, x_max_1], [y_min_1, y_max_1])
ax[1].plot([x_min_2, x_max_2], [y_min_2, y_max_2])

ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              label='class 0', marker='o')
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
              label='class 1', marker='s')

ax[1].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              label='class 0', marker='o')
ax[1].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
              label='class 1', marker='s')

ax[1].legend(loc='lower right')

plt.ylim([-3, 3])
plt.xlim([-3, 3])
plt.show()

# As we can see now, random weight initialization breaks
# the symmetry in the weight updates if we use
# randomly initialized weights
