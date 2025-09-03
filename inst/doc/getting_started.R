## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = reticulate::py_module_available("keras")
)
# Suppress verbose Keras output for the vignette
options(keras.fit_verbose = 0)
set.seed(123)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("pak")
# pak::pak("davidrsch/kerasnip")
# pak::pak("rstudio/keras3")
# 
# # Install the backend
# keras3::install_keras()

## ----load-kerasnip------------------------------------------------------------
library(kerasnip)
library(tidymodels)
library(keras3)

## ----prepare-data-------------------------------------------------------------
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# Rescale
x_train <- x_train / 255
x_test <- x_test / 255

# Convert outcomes to factors for tidymodels
# kerasnip will handle y convertion internally using keras3::to_categorical()
y_train_factor <- factor(y_train)
y_test_factor <- factor(y_test)

# For tidymodels, it's best to work with data frames
# Use I() to keep the matrix structure of x within the data frame
train_df <- data.frame(x = I(x_train), y = y_train_factor)
test_df <- data.frame(x = I(x_test), y = y_test_factor)

## ----keras-standard, eval=FALSE, echo=TRUE, results='hide'--------------------
# # The standard Keras3 approach
# model <- keras_model_sequential(input_shape = 784) |>
#   layer_dense(units = 256, activation = "relu") |>
#   layer_dropout(rate = 0.4) |>
#   layer_dense(units = 128, activation = "relu") |>
#   layer_dropout(rate = 0.3) |>
#   layer_dense(units = 10, activation = "softmax")
# 
# summary(model)
# 
# model |>
#   compile(
#     loss = "categorical_crossentropy",
#     optimizer = optimizer_rmsprop(),
#     metrics = "accuracy"
#   )
# 
# # The model would then be trained with model |> fit(...)

## ----define-blocks------------------------------------------------------------
# An input block to initialize the model.
# The 'model' argument is supplied implicitly by the kerasnip backend.
mlp_input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}

# A reusable "module" that combines a dense layer and a dropout layer.
# All arguments that should be tunable need a default value.
dense_dropout_block <- function(model, units = 128, rate = 0.1) {
  model |>
    layer_dense(units = units, activation = "relu") |>
    layer_dropout(rate = rate)
}

# The output block for classification.
mlp_output_block <- function(model, num_classes) {
  model |> layer_dense(units = num_classes, activation = "softmax")
}

## ----create-spec--------------------------------------------------------------
create_keras_sequential_spec(
  model_name = "mnist_mlp",
  layer_blocks = list(
    input = mlp_input_block,
    hidden_1 = dense_dropout_block,
    hidden_2 = dense_dropout_block,
    output = mlp_output_block
  ),
  mode = "classification"
)

## ----use-spec-----------------------------------------------------------------
mlp_spec <- mnist_mlp(
  hidden_1_units = 256,
  hidden_1_rate = 0.4,
  hidden_2_rate = 0.3,
  hidden_2_units =  128,
  compile_loss = "categorical_crossentropy",
  compile_optimizer = optimizer_rmsprop(),
  compile_metrics = c("accuracy"),
  fit_epochs = 30,
  fit_batch_size = 128,
  fit_validation_split = 0.2
) |>
  set_engine("keras")

# Fit the model
mlp_fit <- fit(mlp_spec, y ~ x, data = train_df)

## ----model-summarize----------------------------------------------------------
mlp_fit |>
  extract_keras_model() |>
  summary()

## ----model-plot, eval=FALSE---------------------------------------------------
# mlp_fit |>
#   extract_keras_model() |>
#   plot(show_shapes = TRUE)

## ----model-fit-history--------------------------------------------------------
mlp_fit |>
  extract_keras_history() |>
  plot()

## ----model-evaluate-----------------------------------------------------------
mlp_fit |> keras_evaluate(x_test, y_test)

## ----model-predict-class------------------------------------------------------
# Predict the class for the first 5 images in the test set
class_preds <- mlp_fit |>
  predict(new_data = head(select(test_df, x)))
class_preds

## ----model-predict-prob-------------------------------------------------------
# Predict probabilities for the first 5 images
prob_preds <- mlp_fit |>
  predict(new_data = head(select(test_df, x)), type = "prob")
prob_preds

## ----model-predict-compare----------------------------------------------------
# Combine predictions with actuals for comparison
comparison <- bind_cols(
  class_preds,
  prob_preds
) |>
  bind_cols(
    head(test_df[, "y", drop = FALSE])
  )
comparison

## ----tune-spec-mnist----------------------------------------------------------
# Define a tunable specification
# We set num_hidden_2 = 0 to disable the second hidden block
# for this tuning example
tune_spec <- mnist_mlp(
  num_hidden_1 = tune(),
  hidden_1_units = tune(),
  hidden_1_rate = tune(),
  num_hidden_2 = 0,
  compile_loss = "categorical_crossentropy",
  compile_optimizer = optimizer_rmsprop(),
  compile_metrics = c("accuracy"),
  fit_epochs = 30,
  fit_batch_size = 128,
  fit_validation_split = 0.2
) |>
  set_engine("keras")

# Create a workflow
tune_wf <- workflow(y ~ x, tune_spec)

## ----create-grid-mnist--------------------------------------------------------
# Define the tuning grid
params <- extract_parameter_set_dials(tune_wf) |>
  update(
    num_hidden_1 = dials::num_terms(c(1, 3)),
    hidden_1_units = dials::hidden_units(c(64, 256)),
    hidden_1_rate = dials::dropout(c(0.2, 0.4))
  )
grid <- grid_regular(params, levels = 3)
grid

## ----run-tuning, cache=TRUE---------------------------------------------------
# Using only the first 100 rows for speed. The real call
# should be: folds <- vfold_cv(train_df, v = 3)
folds <- vfold_cv(train_df[1:100, ], v = 3)

tune_res <- tune_grid(
  tune_wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = FALSE, save_workflow = TRUE)
)

## ----show-best-mnist----------------------------------------------------------
# Show the summary table of the best models
show_best(tune_res, metric = "accuracy")

## ----finalize-best-model------------------------------------------------------
# Select the best hyperparameters
best_hps <- select_best(tune_res, metric = "accuracy")

# Finalize the workflow with the best hyperparameters
final_wf <- finalize_workflow(tune_wf, best_hps)

# Fit the final model on the full training data
final_fit <- fit(final_wf, data = train_df)

## ----inspect-final-model------------------------------------------------------
# Print the model summary
final_fit |>
  extract_fit_parsnip() |>
  extract_keras_model() |>
  summary()

# Plot the training history
final_fit |>
  extract_fit_parsnip() |>
  extract_keras_history() |>
  plot()

