small_lr_params = perceptron_train(X_train, y_train, learning_rate=0.01,
                                   mparams=None, zero_weights=True)

for _ in range(2):
    _ = perceptron_train(X_train, y_train, mparams=small_lr_params)

x_min_small = -2
y_min_small = (-(small_lr_params['weights'][0] * x_min) /
               small_lr_params['weights'][1] -
               (small_lr_params['bias'] / small_lr_params['weights'][1]))

x_max_small = 2
y_max_small = (-(small_lr_params['weights'][0] * x_max) /
               small_lr_params['weights'][1] -
               (small_lr_params['bias'] / small_lr_params['weights'][1]))


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min_small, x_max_small], [y_min_small, y_max_small])

ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              label='class 0', marker='o')
ax[0].scatter(X_train[y_trai == 1, 0], X_train[y_train == 1, 1],
              label='class 1', marker='s')

ax[1].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              label='class 0', marker='o')
ax[1].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
              label='class 1', marker='s')

ax[1].legend(loc='lower right')

plt.ylim([-3, 3])
plt.xlim([-3, 3])
plt.show()


print('Learning=1. rate params:', model_params)
print('Learning=0.01 rate params:', small_lr_params)

# As we can see, lowering the learning rate changes the model parameters.
# But if we look closely, we can see that in the case of the perceptron
# the learning rate is just a scaling factor of the weight & bias vector
# if we initialize the weights to zero. Therefore, the decision region
# is exactly the same for different learning rates.
