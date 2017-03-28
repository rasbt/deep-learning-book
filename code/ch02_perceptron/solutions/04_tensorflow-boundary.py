x_min = -2
y_min = (-(modelparams['weights'][0] * x_min) / modelparams['weights'][1] -
         (modelparams['bias'][0] / model_params['weights'][1]))

x_max = 2
y_max = (-(modelparams['weights'][0] * x_max) / modelparams['weights'][1] -
         (modelparams['bias'][0] / modelparams['weights'][1]))


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])

ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              label='class 0', marker='o')
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
              label='class 1', marker='s')

ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
              label='class 0', marker='o')
ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
              label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()


# The TensorFlow model performs better on the test set just by random chance.
# Remember, the perceptron algorithm stops learning as soon as it classifies
# the training set perfectly.
# Possible explanations why there is a difference between the NumPy and
# TensorFlow outcomes could thus be numerical precision, or slight differences
# in our implementation.
